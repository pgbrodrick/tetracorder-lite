"""
Decoder for the tetracorder "expert system" command file (``cmd.lib.setup.*``).

Tetracorder's expert system file is a plain-text description of every material it
maps: which library spectrum identifies it, the spectral features (continuum
endpoints) that define it, and the fit/depth constraints applied.

The main class :class:`TetraDecoder` parses these expert system files into structured
dictionaries and provides methods to export the data to various formats (CSV, DataFrame).
It automatically resolves variable references from companion files (``cmd.lib.setup.variables``,
``cmds.start.*``) when they are present.

Examples
--------
Basic usage to decode an expert system file:

>>> decoder = TetraDecoder("cmd.lib.setup.t5.51d")
>>> decoder.blocks  # List of decoded material records
>>> df = decoder.table()  # Get as pandas DataFrame

Decode specific groups and export to CSV:

>>> decoder = TetraDecoder("cmd.lib.setup.t5.51d", groups=(1, 2))
>>> decoder.export_csv("output.csv", reference="reference_matrix.csv")

Access decoded records programmatically:

>>> for block in decoder.blocks:
...     print(block["name"], block["library"], block["record"])
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import pandas as pd


Logger = logging.getLogger(__name__)

# Feature importance letter (D=diagnostic, M=must, O=optional, W=weak) + continuum
# shape (L=linear, C=curved) + ``w`` (wavelength units). See tetracorder getifeat.r.
LINEAR_FEATURE_TYPES = ("DLw", "MLw", "OLw", "WLw")
CURVED_FEATURE_TYPES = ("DCw", "MCw", "OCw", "WCw")
FEATURE_TYPES = LINEAR_FEATURE_TYPES + CURVED_FEATURE_TYPES


class TetraDecoder:
    """
    Decode a Tetracorder expert system file into structured per-material records.

    The expert system file (``cmd.lib.setup.<version>``) is organized into "group
    blocks", one per mapped material. Each block names the library spectrum used to
    identify the material, the spectral features (continuum endpoints plus fit/depth
    thresholds) that define it, the constituent constraints applied, and the output
    file the mapper writes. :meth:`decode` walks the file and turns every block whose
    ``group`` is in ``only`` into a plain ``dict`` (see :meth:`parse_block` for the
    schema); the results are stored on :attr:`blocks`.

    Several fields in the file are written as ``[NAME]`` references to variables
    defined in the companion ``cmd.lib.setup.variables`` file, and output paths are
    keyed by group directory in ``cmds.start.*``. When those sidecar files are found
    next to the expert system file they are parsed up front (:attr:`vars`,
    :attr:`groups`) and substituted in during decoding.

    Attributes
    ----------
    file : Path
        Resolved path to the expert system file being decoded.
    vars : dict[str, str] or None
        Variable name -> value map from ``cmd.lib.setup.variables``, or ``None`` if
        that file was not found.
    groups : dict[int, str]
        Group number -> output directory prefix, from ``cmds.start.*``.
    only : tuple[int, ...]
        Group numbers to keep; blocks in other groups go to :attr:`ignored`.
    blocks : list[dict]
        Decoded material records for the kept groups (populated by :meth:`decode`).
    ignored : list[dict]
        Decoded records whose group was not in :attr:`only`.
    """
    def __init__(
        self,
        path: Union[str, Path],
        groups: Tuple[int, ...] = (1, 2),
        decode: bool = True,
        raise_casts: bool = True,
    ):
        """
        Parameters
        ----------
        path : str or Path
            Path to the Tetracorder expert system file (``cmd.lib.setup.<version>``).
            May also be a directory, in which case the first file matching
            ``cmd.lib.setup.t*`` inside it is used.
        groups : tuple[int, ...], optional
            Group numbers to decode. Blocks belonging to any other group are parsed
            but set aside in :attr:`ignored` rather than :attr:`blocks`. Defaults to
            ``(1, 2)`` (the group 1 / group 2 minerals of the EMIT L2B product).
            Pass an empty/falsy value to keep every group found in :attr:`groups`.
        decode : bool, default=True
            Whether to immediately decode the file upon initialization. If False, the
            file can be decoded later by calling :meth:`decode`.
        raise_casts : bool, default=True
            Whether to raise exceptions when type casting fails in :meth:`cast`. If
            False, failed casts return the original value instead of raising an error.
        """
        self.file = Path(path)
        if self.file.is_dir():
            self.file = next(self.file.glob("cmd.lib.setup.t*"))

        # Retrieve external references first, if available
        self.root = self.file.parent

        self.vars = {}
        if (f := self.root / "cmd.lib.setup.variables").exists():
            self.vars = self.parse_variables(f)

        self.groups = {}
        if files := list(self.root.glob(f"cmds.start.*")):
            self.groups = self.parse_group_paths(files[0])

        # NOT-feature reference spectra: ``[NAME]`` -> (library, record). Tetracorder's
        # setup copies one of cmd.lib.setup.nots-ratios-{small,medium,large} into the run
        # directory as cmd.lib.setup.nots-ratios; in the source tree use -small, the
        # default for <= 2171-channel spectrometers.
        self.nots = {}
        for name in ("cmd.lib.setup.nots-ratios", "cmd.lib.setup.nots-ratios-small"):
            if (f := self.root / name).exists():
                self.nots = {
                    key: (lib, int(rec))
                    for lib, entries in self.parse_not_ratios(f).items()
                    for key, rec in entries.items()
                }
                break

        self.only = groups
        if not groups:
            self.only = list(self.groups)

        self._raise = raise_casts

        if decode:
            self.decode()

    def decode(self) -> List[dict]:
        """
        Read the expert system file and parse every group block.

        The file is read line by line into :attr:`lines`, split into blocks by
        :meth:`extract_blocks`, and each block is parsed by :meth:`parse_block`.
        Parsed blocks are sorted into :attr:`blocks` (group in :attr:`only`) or
        :attr:`ignored` (any other group).

        Returns
        -------
        list[dict]
            The kept blocks, i.e. :attr:`blocks`.
        """
        self.lines = []
        with self.file.open("r") as file:
            for line in file:
                line = line.strip()
                self.lines.append(line)


        self.blocks = []
        self.ignored = []
        for block in self.extract_blocks(self.lines):
            block = self.parse_block(block)
            if block["group"] in self.only:
                self.blocks.append(block)
            else:
                self.ignored.append(block)

        return self.blocks

    def parse_block(self, block: List[str]) -> dict:
        """
        Parse a single group block into a material record.

        Each block is a list of stripped, comment-free lines (as produced by
        :meth:`extract_blocks`). Recognized directives are pulled out into a ``dict``;
        any ``[NAME]`` references are substituted with their values from :attr:`vars`.

        Parameters
        ----------
        block : list[str]
            Lines of one group block from the expert system file.

        Returns
        -------
        dict
            A material record with any of the following keys, depending on which
            directives the block contains:

            ``group`` : int
                Group number the material belongs to.
            ``use`` : str
                Whether the entry is active (``"yes"``/``"no"``).
            ``name`` : str
                Short material name (first token after ``TITLE=``).
            ``title`` : str
                Full material title.
            ``library`` : str
                Spectral library ID the reference record lives in (e.g. ``splib06``).
            ``record`` : int
                Record number of the reference spectrum within ``library``.
            ``materials`` : list[str]
                Spectrally active components from ``list materials`` (e.g.
                ``neodymium``, ``kaolinite``).
            ``filename`` : str
                Output base file name written by the mapper.
            ``path`` : str
                ``filename`` prefixed with its group output directory, when the group
                directory is known from :attr:`groups`.
            ``data_type_scaling`` : float
                Scaling applied to the 8-bit (``0-255``) output DN.
            ``features`` : list[dict]
                Spectral features. Each feature has ``feature_type`` (linear
                ``DLw``/``MLw``/``OLw``/``WLw`` or curved ``DCw``/``MCw``/``OCw``/``WCw``),
                a ``continuum`` of window endpoints (4 values for linear, 8 for
                curved continua), and
                one list per fit/depth constraint keyword found on the feature line
                (e.g. ``ct``, ``rcbblc<``).
            ``not_features`` : list[dict]
                Exclusion features (``n... NOT`` lines); see :meth:`parse_not`.
            ``constituent_constraints`` : dict[str, list[float]]
                Per-block constraint keyword -> values (e.g. ``FD-FIT>``, ``FITALL>``).
        """
        data = {}
        for i, line in enumerate(block):
            if line.startswith("group"):
                data["group"] = int(line.split()[1])

            elif line.startswith("use="):
                data["use"] = line.split(" ")[1]

            elif line.startswith("ID="):
                data["id"] = line[3:].strip()

            elif "=- TITLE=" in line:
                split = line.split("TITLE=")[1].split()
                data["name"] = split[0]
                data["title"] = " ".join(split)

            elif line.startswith("define library records"):
                split = block[i+1].split()
                data["library"] = split[2].strip("[]")
                data["record"] = int(split[3])

            elif line.startswith("list materials"):
                # Spectrally active components, one per line (first token is the name)
                data["materials"] = []
                for mat in block[i + 1:]:
                    if mat.startswith("end_list_materials"):
                        break
                    data["materials"].append(mat.split()[0])

            elif line.startswith("define output"):
                data["filename"] = block[i+2]
                data["path"] = None
                if path := self.groups.get(data["group"]):
                    data["path"] = path + data["filename"]

            elif line.startswith("8 DN 255"):
                val = line.split(" ")[-1]
                val = self.vars.get(val.strip("[]"), val) # Substitute variables, if necessary
                data["data_type_scaling"] = self.cast(float, val)

            elif line.startswith("define features"):
                data["features"] = []
                by_label = {}
                while (line := block[i]) != "endfeatures":
                    i += 1
                    # Substitute variables, if necessary. A variable may hold several
                    # numbers (e.g. [RBD22] = "0.0007 0.0011"), so re-split afterwards.
                    split = " ".join(self.vars.get(v.strip("[]"), v) for v in line.split()).split()
                    if line.startswith("n") and len(split) > 1 and split[1] == "NOT":
                        data.setdefault("not_features", []).append(self.parse_not(line))
                    elif line.startswith("f") and len(split) > 1 and split[1] in FEATURE_TYPES:
                        # Linear continua (xLw) give 4 endpoints; curved (xCw) give 8
                        n = 4 if split[1] in LINEAR_FEATURE_TYPES else 8
                        feat = {
                            "feature_type": split[1],
                            "continuum": self.cast(float, split[2:2 + n]),
                        }
                        feat.update(self.parse_feature_constraints(split[2 + n:]))
                        by_label[split[0]] = feat
                        data["features"].append(feat)
                    elif line.startswith("f") and len(split) > 1 and split[0] in by_label:
                        # Continuation line ("f1a   r*bd> ..."): more constraints for the
                        # feature already declared under the same label.
                        by_label[split[0]].update(self.parse_feature_constraints(split[1:]))

            elif line.startswith("constraint:"):
                consts = data.setdefault("constituent_constraints", {})
                # Match successive KEY>VALUE pairs, where VALUE extends until the next KEY> or end of line
                matches = re.findall(r"([^\s>]+)>\s*(.*?)(?=\s+[^\s>]+>|$)", line)
                for key, val in matches:
                    val = self.vars.get(val.strip("[]"), val)
                    consts[key] = self.cast(float, val.split())

        return data

    @staticmethod
    def parse_feature_constraints(tokens: List[str]) -> Dict[str, List[float]]:
        """
        Parse per-feature constraint tokens into ``{key: [values]}``.

        Keys are the non-numeric tokens (``ct``, ``r*bd>``, ``lct/rct>``, ``rcbblc<``,
        ...); the numbers that follow a key are its values. ``weight*2.2`` is a single
        token and becomes ``{"weight": [2.2]}``. Variables must already be substituted.
        """
        out: Dict[str, List[float]] = {}
        key = None
        for tok in tokens:
            if tok.startswith("weight*"):
                try:
                    out["weight"] = [float(tok.split("*", 1)[1])]
                except ValueError:
                    out["weight"] = [tok.split("*", 1)[1]]
                key = None
                continue
            try:
                val = float(tok)
            except ValueError:
                key = tok
                out.setdefault(key, [])
                continue
            if key is not None:
                out[key].append(val)
        return out

    def parse_not(self, line: str) -> dict:
        """
        Parse an exclusion (``NOT``) feature line.

        Format (tetracorder ``getnotfeat``)::

            n1a NOT [NOTGREENVEG]  1  0.50a 0.5
            n1a NOT [NOTGREENVEG]  2  0.10r1 0.5

        i.e. reference material (a ``[NAME]`` from ``cmd.lib.setup.nots-ratios`` or an
        explicit ``library record`` pair), the feature number *of that material*, a
        depth threshold flagged ``a`` (absolute) or ``rN`` (relative to this material's
        feature ``N``), and a fit threshold. Tetracorder zeroes this material's
        fit/depth wherever the referenced feature is found with depth and fit above
        both thresholds.

        Returns
        -------
        dict
            ``ref`` (the name or ``"lib rec"``), ``library``, ``record``, ``feature``
            (1-based, of the referenced material), ``depth``, ``relative_to`` (``None``
            for absolute, else this material's 1-based feature number), ``fit``.
        """
        tokens = line.split("\\#")[0].split()[2:]
        ref = tokens[0]
        if ref.startswith("["):
            name = ref.strip("[]")
            lib, rec = self.nots.get(name, (None, None))
            rest = tokens[1:]
        else:                                  # explicit "<lib-letter> <record>"
            name, lib, rec = f"{tokens[0]} {tokens[1]}", tokens[0], self.cast(int, tokens[1])
            rest = tokens[2:]
        m = re.match(r"^([0-9.]+)([ar])(\d*)$", rest[1])
        if not m:
            raise ValueError(f"Unparseable NOT depth threshold {rest[1]!r} in {line!r}")
        return {
            "ref": name,
            "library": lib,
            "record": rec,
            "feature": int(rest[0]),
            "depth": float(m.group(1)),
            "relative_to": int(m.group(3)) if m.group(2) == "r" else None,
            "fit": float(rest[2]),
        }

    def get_groups(self, groups: Iterable[int]) -> List[dict]:
        """
        Return the decoded records belonging to the given groups.

        Filters :attr:`blocks` (the kept records) by group number.

        Parameters
        ----------
        groups : Iterable[int]
            Group numbers to select.

        Returns
        -------
        list[dict]
            Records from :attr:`blocks` whose ``group`` is in ``groups``. See
            :meth:`parse_block` for the per-record schema.
        """
        return [block for block in self.blocks if block["group"] in groups]

    def cast(self, dtype: type, val: Any) -> Any:
        """
        Cast a value (or each item of a list) to ``dtype``, tolerating failures.

        Applies ``dtype`` to ``val`` (or elementwise if ``val`` is a list). When a
        conversion fails, the original value is kept unless ``raise_casts`` was set on
        the decoder, in which case the error propagates. Used to coerce the numeric
        fields parsed out of the expert system while leaving unparseable tokens (e.g.
        unresolved ``[NAME]`` references) intact.

        Parameters
        ----------
        dtype : type
            The target type/callable applied to each value (e.g. ``float``).
        val : Any
            A single value or a list of values to cast.

        Returns
        -------
        Any
            The cast value, or a list of cast values when ``val`` is a list.
        """
        def apply(v: Any) -> Any:
            """Apply dtype conversion to a single value, tolerating failures."""
            try:
                return dtype(v)
            except:
                if self._raise:
                    raise
                return v

        if isinstance(val, list):
            return [apply(v) for v in val]
        return apply(val)

    def table(self,
        groups: Optional[Iterable[int]] = None,
        columns: Tuple[str, ...] = ("id", "group", "library", "record", "title", "path"),
        pandas: bool = True
    ) -> Union[pd.DataFrame, List[list]]:
        """
        Collect the decoded records into a tabular form, one row per material.

        Selects one row per material across both :attr:`blocks` and :attr:`ignored`
        whose group is in ``groups`` (so records outside :attr:`only` can be included
        by widening ``groups``), keeping only the fields named in ``columns``.

        Parameters
        ----------
        groups : Iterable[int], optional
            Group numbers to include. Defaults to :attr:`only` (the groups that were
            decoded into :attr:`blocks`).
        columns : tuple[str, ...], optional
            Record keys to emit as columns, in order. Every named key must be present
            on each included record (see :meth:`parse_block` for available keys).
            Defaults to ``("group", "library", "record", "title", "path")``.
        pandas : bool, default=True
            If True, return a :class:`pandas.DataFrame`. If False, return a list of
            rows with ``columns`` as the leading header row.

        Returns
        -------
        pandas.DataFrame or list[list]
            A DataFrame of the selected records when ``pandas`` is True; otherwise a
            list of rows whose first element is the ``columns`` header.
        """
        if groups is None:
            groups = self.only

        data = [columns]
        for block in self.blocks + self.ignored:
            if block["group"] in groups:
                data.append([block[c] for c in columns])

        if pandas:
            return pd.DataFrame(data[1:], columns=columns)
        return data

    def match_ref(
        self,
        matrix: Union[str, Path],
        nu: Optional[pd.DataFrame] = None,
        keys: list[str] = ["id"],
        sortby: str = None,
        clean_titles: bool = False,
    ) -> pd.DataFrame:
        """
        Align the decoded records to a reference matrix's stable ``index`` column.

        Merges the ``index`` column from an existing reference matrix CSV onto the
        decoded table, matching on ``record`` / ``library`` so each material keeps the
        index it had in the reference. Records with no match in the reference are
        treated as new and assigned fresh indices continuing past the reference's max,
        and the additions are logged.

        Parameters
        ----------
        matrix : str or Path
            Path to the reference matrix CSV. It must contain ``record``, ``library``,
            and ``index`` columns (column names are matched case-insensitively).
        nu : pandas.DataFrame, optional
            The table to align. Defaults to :meth:`table` (the decoded records). Must
            contain ``record`` and ``library`` columns.
        keys : list[str], default=["record", "library", "group"]
            Column names to use as the merge key when matching records between the
            reference matrix and the new table.
        sortby : str, optional
            Column name to sort the output by. If None (default), the order is
            preserved from the input.
        clean_titles : bool, default=False
            Whether to remove family identifiers (substrings containing ``=``) from
            the ``title`` column.

        Returns
        -------
        pandas.DataFrame
            ``nu`` with a leading ``index`` column carrying the reference indices, and
            newly-assigned indices for records absent from the reference.
        """
        og = pd.read_csv(matrix)

        if nu is None:
            nu = self.table()

        # Backwards compatibility that used Capitalized columns
        og = og.rename(columns={c: c.lower() for c in og})

        columns = keys + ["index"]
        if "url" in og:
            columns.append("url")

        # Merge using record/library
        nu = nu.merge(
            og[columns],
            on=keys,
            how="left",
        )
        nu["index"] = nu["index"].astype("Int64")

        # Reorder index to the first column
        nu.insert(0, "index", nu.pop("index"))

        # Update index of missing entries
        nul = nu["index"].isnull()

        off = nu["index"].max() + 1
        rng = range(off, off + nul.sum())

        if sortby:
            nu = nu.sort_values(by=sortby)

        if clean_titles:
            # Remove "family" identifiers from title strings (if the substring has an `=`)
            nu["title"] = nu["title"].str.replace(r"\s*\S*=[^\s]*", "", regex=True)

        new = nu[nul]
        fmt = new.set_index("index").to_string(index=False)
        Logger.info(f"{len(new)} records are new:\n{fmt}")
        Logger.info(f"Setting these as {rng}")

        nu.loc[nul, "index"] = rng

        return nu

    def export_csv(
        self,
        file: str,
        groups: Optional[Iterable[int]] = None,
        columns: Tuple[str, ...] = ("id", "group", "library", "record", "title", "path"),
        reference: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> None:
        """
        Write the decoded records to a CSV file, one row per material.

        Emits one row per material across both :attr:`blocks` and :attr:`ignored`
        whose group is in ``groups``, so records outside :attr:`only` can be exported
        by widening ``groups``.

        Parameters
        ----------
        file : str
            Destination path for the CSV file.
        groups : Iterable[int], optional
            Group numbers to include. Defaults to :attr:`only` (the groups that were
            decoded into :attr:`blocks`).
        columns : tuple[str, ...], optional
            Record keys to emit as columns, in order. Every named key must be present
            on each included record (see :meth:`parse_block` for available keys).
            Defaults to ``("group", "library", "record", "title", "path")``.
        reference : str or Path, optional
            Path to a reference matrix CSV. When given, a stable ``index`` column is
            added by aligning to it via :meth:`match_ref` (matching on
            ``record`` / ``library``).
        **kwargs
            Additional keyword arguments passed to :meth:`match_ref` (e.g. ``sortby``,
            ``clean_titles``).

        Returns
        -------
        None
            Writes output to the file specified by ``file``. No value is returned.
        """
        df = self.table(groups, columns, pandas=True)

        # Unique keys to use for matching a reference matrix to a new one
        keys = ["id"]
        dups = df[df.duplicated(subset=keys, keep=False)]
        if not dups.empty:
            Logger.warning(f"Found {len(dups)} duplicates using keys {keys}:\n{dups.to_string(index=False)}")

        if reference:
            df = self.match_ref(reference, nu=df, keys=keys, **kwargs)

        df.to_csv(file, index=False)

    @staticmethod
    def parse_variables(file: str) -> Dict[str, str]:
        """
        Parse ==[NAME] value... definitions from a command file.

        This function extracts variable definitions from tetracorder command files,
        which use a special ==[NAME] syntax to define parameters. The values are
        returned as strings to preserve formatting and precision.

        Parameters
        ----------
        file : str
            Path to the command file to parse.

        Returns
        -------
        Dict[str, str]
            Mapping of variable names to their string values (whitespace-stripped).
            If a variable is defined multiple times, the last definition wins.

        Examples
        --------
        Given a file with content:
            ==[THRESHOLD] 0.5
            ==[VALUES] 1.0 2.0 3.0
            ==[THRESHOLD] 0.8

        >>> parse_variables("cmd.file")
        {'THRESHOLD': '0.8', 'VALUES': '1.0 2.0 3.0'}
        """
        text = Path(file).read_text()

        pattern = re.compile(
            r"^==\[(?P<name>[^\]]+)\]\s*(?P<values>[-+0-9.eE\s]+)",
            re.MULTILINE,
        )

        return {
            match["name"]: match["values"].strip()
            for match in pattern.finditer(text)
        }

    @staticmethod
    def parse_not_ratios(file: str) -> Dict[str, Dict[str, str]]:
        """
        Parse the ``cmd.lib.setup.nots-ratios`` "not feature" definitions.

        These entries use a ``==[NAME] [SOURCE] value...`` syntax, grouping variable
        definitions under a source library key (e.g. ``rec1``, ``Bckm``). Values are
        kept as whitespace-stripped strings.

        Parameters
        ----------
        file : str
            Path to the ``cmd.lib.setup.nots-ratios`` file.

        Returns
        -------
        dict[str, dict[str, str]]
            Nested mapping ``{source: {name: value}}``. If a name is repeated within
            a source, the last definition wins.
        """
        text = Path(file).read_text()

        pattern = re.compile(
            r"^==\[(?P<name>[^\]]+)\]\s*\[(?P<source>[\w]+)\]\s*(?P<values>[-+0-9.eE\s]+)",
            re.MULTILINE,
        )

        data = {}
        for m in pattern.finditer(text):
            source = data.setdefault(m["source"], {})
            source[m["name"]] = m["values"].strip()

        return data

    @staticmethod
    def parse_group_paths(file: str) -> Dict[int, str]:
        """
        Parse the per-group output directories from a ``cmds.start.*`` file.

        Group directories are defined with a ``==[DIRg<N>]<path>`` syntax, e.g.
        ``==[DIRg1]group.1um/``. These prefixes are prepended to each block's output
        filename in :meth:`parse_block`.

        Parameters
        ----------
        file : str
            Path to the ``cmds.start.*`` file.

        Returns
        -------
        Dict[int, str]
            Mapping of group number to its output directory prefix.
        """
        text = Path(file).read_text()

        matches = re.findall(r"^==\[DIRg(\d+)\](.+)$", text, flags=re.MULTILINE)

        return {int(group): path for group, path in matches}

    @staticmethod
    def extract_blocks(lines: Iterable[str]) -> Iterator[List[str]]:
        """
        Split the expert system lines into group blocks.

        A block opens on a line matching ``group <N>`` and closes on the line
        starting with ``endaction``. Within a block, trailing ``\\#`` comments are
        stripped (except on ``TITLE`` lines, where ``#`` may be significant) and blank
        lines are dropped, so each yielded block is a list of meaningful lines only.

        Parameters
        ----------
        lines : Iterable[str]
            Stripped lines of the expert system file.

        Yields
        ------
        list[str]
            One group block, from its ``group <N>`` line through ``endaction``.
        """
        block = None
        for line in lines:
            if re.match(r"^group\s+\d", line):
                block = []

            if block is not None:
                # Strip comments from the line
                if "TITLE" not in line:
                    line = line.split("\\#")[0].strip()

                if line:
                    block.append(line)
                    if line.startswith("endaction"):
                        yield block
                        block = None
