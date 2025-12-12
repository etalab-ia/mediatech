import json
import os
import string
import unicodedata

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from config import (
    SERVICE_PUBLIC_PART_DATA_FOLDER,
    SERVICE_PUBLIC_PRO_DATA_FOLDER,
    TRAVAIL_EMPLOI_DATA_FOLDER,
    get_logger,
)

logger = get_logger(__name__)

### Imported functions from the pyalbert library

# *********
# * Utils *
# *********


def normalize(text: str) -> str:
    # Like removing non-breaking space in latin-1 (\xa0)
    return unicodedata.normalize("NFKC", text)


def get_text(obj):
    if isinstance(obj, NavigableString):
        t = obj.string.strip()
    else:
        t = obj.get_text(" ", strip=True)
    return normalize(t)


def extract(soup: Tag, tag: str, pop=True, recursive=True) -> str:
    t = soup.find(tag, recursive=recursive)
    if not t:
        return ""

    if pop:
        t.extract()

    return get_text(t)


def extract_all(soup: Tag, tag: str, pop=True) -> list[str]:
    if not soup.find(tag):
        return []

    elts = []
    for t in soup.find_all(tag):
        if pop:
            t.extract()
        elts.append(get_text(t))

    return elts


# ***************
# * Sheet parsing *
# ***************


def _get_xml_files_paths(path: str) -> list[str]:
    """
    Returns sorted list of XML files paths.

    If `path` is a file returns [path]. If a directory, walk it and include files
    that end with '.xml' and start with 'N' or 'F'.
    """
    xml_files = []

    if os.path.isfile(path):
        # Use to test result on a particular file
        xml_files = [path]
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".xml") and file.startswith(("N", "F")):
                    # Keep only "fiches pratiques", "fiches questions-réponses",
                    # "fiches thème", "fiches dossier".
                    fullpath = os.path.join(root, file)
                    xml_files.append(fullpath)
    return sorted(xml_files)


def _get_metadata(soup) -> dict:
    """
    Extract metadata fields from a BeautifulSoup-parsed document.

    Parameters:
        soup (bs4.BeautifulSoup): Parsed XML/HTML document containing publication metadata.

    Returns:
        dict: Metadata with keys:
            - url (str): Publication 'spUrl' attribute if present, else empty string.
            - audience (str): Comma-separated values from all 'Audience' elements.
            - theme (str): Comma-separated values from all 'Theme' elements.
            - surtitle (str): Value from 'SurTitre'.
            - subject (str): Value from 'dc:subject'.
            - title (str): Value from 'dc:title'.
            - description (str): Value from 'dc:description'.
            - introduction (str): Value from 'Introduction' or falls back to description if empty.
    """
    url = ""
    if soup.find("Publication") is not None:
        if "spUrl" in soup.find("Publication").attrs:
            url = soup.find("Publication")["spUrl"]

    doc = {
        "url": url,
        "audience": ", ".join(extract_all(soup, "Audience")),
        "theme": ", ".join(extract_all(soup, "Theme")),
        "surtitle": extract(soup, "SurTitre"),
        "subject": extract(soup, "dc:subject"),
        "title": extract(soup, "dc:title"),
        "description": extract(soup, "dc:description"),
        "introduction": extract(soup, "Introduction"),
    }
    if not doc["introduction"]:
        doc["introduction"] = doc["description"]

    return doc


def _table_to_md(soup):
    # FIXME: multi-column header are lost
    #        e.g.: https://entreprendre.service-public.fr/vosdroits/F3094
    md_table = ""

    # Headers
    for row in soup.find_all("Rangée", {"type": "header"}):
        md_table += "| "
        for cell in row.find_all("Cellule"):
            md_table += get_text(cell) + " | "
        md_table += "\n"

        # Separator
        md_table += "| "
        for _ in row.find_all("Cellule"):
            md_table += "- | "
        md_table += "\n"

    # Rows
    for row in soup.find_all("Rangée", {"type": "normal"}):
        md_table += "| "
        for cell in row.find_all("Cellule"):
            md_table += get_text(cell) + " | "
        md_table += "\n"

    return md_table


def _not_punctuation(s):
    return s.strip()[-1:] not in string.punctuation


def _parse_xml_text_structured(
    current: list[str], context: list[str], soups: list[Tag], recurse=True, depth=0
) -> list[dict]:
    """
    Recursively parse a Service Public XML document into structured chunks.

    Splits XML content into distinct sections while preserving the hierarchical
    context (list of parent titles). Creates a new chunk whenever it
    encounters a <Situation>, <Chapitre>, or <SousChapitre> element.

    Args:
        current: Accumulated text fragments for the chunk being built.
        context: List of parent titles (e.g., ["Enfant majeur", "Vit chez vous"]).
        soups: XML nodes to traverse.
        recurse: If True, continue creating sub-chunks for nested sections.
        depth: Current recursion depth (0 = root level).

    Returns:
        List of dictionaries, each containing:
            - "text": Chunk content (str if depth=0, list[str] otherwise)
            - "context": List of parent titles for this chunk

    Element Handling:
        - <Situation>, <Chapitre>: Create new chunk, add title to context, recurse
        - <SousChapitre>: Create new chunk but stop further recursion
        - <BlocCas>: Format as "Case {title}: {content}"
        - <Liste>: Format as "- {item}"
        - <Tableau>: Convert to Markdown table
        - <ANoter>, <ASavoir>, <Attention>, <Rappel>: Format as "(Title: content)"

    Example:
        Input XML:
            <Situation>
                <Titre>Enfant mineur</Titre>
                <Texte>Vous pouvez déduire...</Texte>
            </Situation>

        Output:
            [{"text": "Vous pouvez déduire...", "context": ["Enfant mineur"]}]
    """
    # TODO:
    #  - could probably be optimized by not extracting tag, just reading it; see extract()
    #  - <TitreAlternatif> can be present next to title
    #  - extract text and reference law (legifrance.fr); see <Reference>
    state = []

    for part in soups:
        if isinstance(part, NavigableString):
            current.append(get_text(part))
            # New chunk
            state.append({"text": current, "context": context})
            current = []
            continue

        for child in part.children:
            # Purge current
            current = [c for c in current if c]

            if isinstance(child, NavigableString):
                current.append(get_text(child))
                continue

            if child.name in ("Situation", "Chapitre", "SousChapitre") and recurse:
                if child.name in ("Situation", "Chapitre"):
                    new_recurse = True
                else:  # "SousChapitre"
                    new_recurse = False

                # New chunk
                if current:
                    state.append({"text": current, "context": context})
                    current = []

                title = extract(child, "Titre", recursive=False)
                if title:
                    new_context = context + [title]
                else:
                    new_context = context

                s = _parse_xml_text_structured(
                    [], new_context, [child], recurse=new_recurse, depth=depth + 1
                )
                state.extend(s)

            elif child.name in ("BlocCas", "Liste"):
                scn = "Cas" if child.name == "BlocCas" else "Item"
                blocs = "\n"
                for subchild in child.children:
                    if subchild.name != scn:
                        if subchild.string.strip():
                            logger.debug(f"XML warning: {child.name} has orphan text")
                        continue

                    if child.name == "BlocCas":
                        title = extract(subchild, "Titre", recursive=False)
                    s = _parse_xml_text_structured(
                        [], context, [subchild], recurse=False, depth=depth + 1
                    )
                    content = " ".join([" ".join(x["text"]) for x in s]).strip()

                    if child.name == "BlocCas":
                        blocs += f"Cas {title}: {content}\n"
                    else:  # Liste
                        blocs += f"- {content}\n"

                current.append(blocs)

            elif child.name == "Tableau":
                title = extract(child, "Titre", recursive=False)
                sep = ":" if _not_punctuation(title) else ""
                table = _table_to_md(child)
                table = f"\n{title}{sep}\n{table}"
                current.append(table)

            elif child.name in ("ANoter", "ASavoir", "Attention", "Rappel"):
                title = extract(child, "Titre", recursive=False)
                sep = ": " if title else ""
                current.append(f"({title}{sep}{get_text(child)})\n")

            elif child.name == "Titre":
                # Format title inside chunks (sub-sub sections etc)
                title = get_text(child)
                sep = ":" if _not_punctuation(title) else ""
                current.append(f"{title}{sep}")

            else:
                # Space joins
                s = _parse_xml_text_structured(
                    current, context, [child], recurse=recurse, depth=depth + 1
                )
                current = []
                sub_state = []
                for x in s:
                    if len(x["context"]) == len(context):
                        # Same context
                        if child.name in (
                            "Chapitre",
                            "SousChapitre",
                            "Cas",
                            "Situation",
                        ):
                            # Add a separator
                            x["text"][-1] += "\n"
                        elif (
                            child.name in ("Paragraphe",)
                            and child.parent.name not in ("Item", "Cellule")
                            and _not_punctuation(x["text"][-1])
                        ):
                            # Title !
                            x["text"][-1] += ":"

                        current.extend(x["text"])
                    else:
                        # New chunk
                        sub_state.append(x)

                if sub_state:
                    state.append({"text": current, "context": context})
                    current = []
                    state.extend(sub_state)

        # New chunk
        if current:
            state.append({"text": current, "context": context})
            current = []

    if depth == 0:
        state = [d for d in state if "".join(d["text"])]
        punctuations = (".", ",", ";")
        for d in state:
            texts = ""
            for i, x in enumerate(d["text"]):
                if not x:
                    continue

                if i > 0 and x.startswith(punctuations):
                    # Stretch join / fix punctuations extra space
                    texts += x
                elif x.startswith("\n") or texts.endswith("\n"):
                    # Strech join / do not surround newline with space
                    texts += x
                else:
                    # Space join
                    texts += " " + x

            d["text"] = texts.strip()

    return state


def _parse_xml_text(xml_file_path: str, structured: bool = False) -> dict:
    """
    Parse a service-public XML file and return a normalized dictionary containing document metadata and content.

    Args:
        xml_file_path (str): Path to the XML file to parse.
        structured (bool, optional): If True, preserves document sections and structure.
            If False, returns flattened text content. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - Metadata fields (url, sid, source, introduction, etc.) via _get_metadata
            - sid (str): Service identifier extracted from URL
            - source (str): Always set to "service-public"
            - related_questions (list[dict]): List of related Q&A with keys:
                - question (str): Question text
                - sid (str): Question identifier
                - url (str): Full URL to the question
            - web_services (list[dict]): List of online services with keys:
                - title (str): Service title
                - institution (str): Institution providing the service
                - url (str): Service URL
                - type (str): Service type
            - text (list[dict] | list[str]): Document content. Format depends on `structured`:
                - If structured=True: List of dicts with structured sections
                - If structured=False: List containing single flattened text string

    Notes:
        - Removes noise elements
        - Normalizes all text content
        - Deduplicates related questions and web services by title/question text
        - When structured=True and sections exist, appends section list to introduction
    """
    with open(xml_file_path, mode="r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    doc = _get_metadata(soup=soup)  # Initialize doc with metadata
    doc["sid"] = doc["url"].split("/")[-1]
    doc["source"] = "service-public"

    # Clean document / Remove potential noise
    extract_all(soup, "OuSAdresser")
    extract_all(soup, "RefActualite")

    def drop_duplicates(data: list[dict], k: str):
        seen = []
        keeps = []
        for x in data:
            if x[k] in seen:
                continue

            keeps.append(x)
            seen.append(x[k])

        return keeps

    def sp_url_encoder(sid, audience):
        audience_to_uri = {
            "Associations": "associations",
            "Particuliers": "particuliers",
            "Professionnels": "professionnels-entreprises",
        }
        # Do not fail silently
        audience_uri = audience_to_uri[audience]
        return f"https://www.service-public.fr/{audience_uri}/vosdroits/{sid}"

    # Get related questions
    questions = [
        {
            "question": get_text(q),
            "sid": q["ID"],
            "url": sp_url_encoder(q["ID"], q["audience"]),
        }
        for q in soup.find_all("QuestionReponse")
    ]
    questions = drop_duplicates(questions, "question")
    doc["related_questions"] = questions

    # Get the Service/contact ressources
    web_services = []
    for q in soup.find_all("ServiceEnLigne"):
        if not q.get("URL"):
            continue

        title_tag = q.find("Titre")
        source_tag = q.find("Source")

        if not title_tag:
            continue

        service = {
            "title": normalize(title_tag.get_text(" ", strip=True)),
            "institution": normalize(source_tag.get_text(" ", strip=True))
            if source_tag
            else "",
            "url": q["URL"],
            "type": q["type"],
        }
        web_services.append(service)

    web_services = drop_duplicates(web_services, "title")
    doc["web_services"] = web_services

    # Clean document / Remove potential noise
    extract_all(soup, "OuSAdresser")
    extract_all(soup, "ServiceEnLigne")
    extract_all(soup, "QuestionReponse")
    extract_all(soup, "RefActualite")

    # Get all textual content
    # --
    # Introduction
    current = [doc["introduction"]]
    if structured:
        # Save sections for later (de-duplicate keeping order)
        sections = [
            get_text(a.Titre)
            for a in soup.find_all("Chapitre")
            if a.Titre and not a.find_parent("Chapitre")
        ]
        sections = list(dict.fromkeys(sections))

        context = []
        top_list = []
        for x in soup.find_all("Publication"):
            for obj in x.children:
                if obj.name in ("Texte", "ListeSituations"):
                    top_list.append(obj)
        texts = _parse_xml_text_structured(
            current=current, context=context, soups=top_list
        )

        if texts and sections:
            # Add all sections title at the end of the introduction
            sections = "\n".join(f"- {section}" for section in sections)
            sections = (
                "\n\nVoici une liste de différentes questions ou thématiques relatives à ce sujet :\n"
                + sections
            )
            texts[0]["text"] += sections

    else:
        if soup.find("ListeSituations") is not None:
            current.append("Liste des situations :")
            for i, situation in enumerate(
                soup.find("ListeSituations").find_all("Situation")
            ):
                if not situation.find("Titre"):
                    logger.debug("warning: Situation > Titre, not found")
                    continue

                if not situation.find("Texte"):
                    logger.debug("warning: Situation > Texte, not found")
                    continue

                situation_title = normalize(
                    situation.find("Titre").get_text(" ", strip=True)
                )
                situation_texte = normalize(
                    situation.find("Texte").get_text(" ", strip=True)
                )
                current.append(f"Cas n°{i + 1} : {situation_title} : {situation_texte}")

        if soup.find("Publication") is not None:
            t = soup.find("Publication").find("Texte", recursive=False)
            if t is not None:
                current.append(normalize(t.get_text(" ", strip=True)))

        texts = [" ".join(current)]

    doc["text"] = texts
    return doc


def _parse_xml_questions(xml_file_path: str) -> list[dict]:
    """
    Parse XML file containing questions and their metadata from Service Public.

    This function extracts question data from an XML file containing either
    QuestionReponse or CommentFaireSi tags and returns a structured list of
    question dictionaries.

    Args:
        xml_file_path (str): Path to the XML file to parse.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - question (str): The question text or formatted question
            - url (str): The Service Public URL for the question
            - tag (str): The XML tag type ("QuestionReponse" or "CommentFaireSi")
    """
    with open(xml_file_path, mode="r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    docs = []
    tags = (
        ("QuestionReponse", lambda x: get_text(x)),
        ("CommentFaireSi", lambda x: f"Comment faire si {get_text(x)} ?"),
    )
    for tag, f in tags:
        for t in soup.find_all(tag):
            audience = t["audience"].lower()
            doc = {
                "question": f(t),
                "url": f"https://www.service-public.fr/{audience}/vosdroits/{t['ID']}",
                "tag": tag,
            }
            docs.append(doc)

    return docs


def _parse_xml(
    target_dir: str, parse_type: str, structured: bool = False
) -> list[dict]:
    """
    Parse XML files from a target directory and extract content based on the specified type.

    Args:
        target_dir (str): Path to the directory containing XML files to parse.
        parse_type (str): Type of parsing to perform. Must be either "text" or "questions".
            - "text": Parse XML files as text documents
            - "questions": Parse XML files as question documents
        structured (bool, optional): If True, parse text in a structured format.
            Only applicable when parse_type is "text". Defaults to False.

    Returns:
        list[dict]: A list of dictionaries containing the parsed content from the XML files.
            - When parse_type is "text": Each dict represents a single document
            - When parse_type is "questions": Each dict represents a single question

    Raises:
        ValueError: If parse_type is not "text" or "questions".
        FileNotFoundError: If the target_dir does not exist.
    """
    if parse_type not in ("text", "questions"):
        raise ValueError()

    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"path {target_dir} to xml sheets not found.")

    xml_files_paths = _get_xml_files_paths(path=target_dir)

    docs = []
    current_pct = 0
    n = len(xml_files_paths)
    for i, xml_file_path in enumerate(xml_files_paths):
        pct = (100 * i) // n
        if pct > current_pct:
            current_pct = pct
            logger.debug(f"Processing sheet: {current_pct}%\r", end="")

        if parse_type == "text":
            doc = _parse_xml_text(
                xml_file_path=xml_file_path, structured=structured
            )  # returns dict
            if doc:
                docs.append(doc)
        elif parse_type == "questions":
            _docs = _parse_xml_questions(xml_file_path=xml_file_path)
            docs.extend(_docs)

    return docs  # list of dicts


def _parse_travailEmploi(target_dir: str, structured: bool = False) -> list[dict]:
    """
    Parse Travail-Emploi data from a JSON file into a structured format.

    Args:
        target_dir (str): Path to the JSON file containing Travail-Emploi data.
        structured (bool, optional): If True, returns text sections as a list of dictionaries
            with normalized text and context (title). If False, joins all sections into a
            single normalized string. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a document with
            the following keys:
            - title (str): Normalized title of the document
            - url (str): URL of the document
            - date (str): Publication date
            - sid (str): Publication ID
            - introduction (str): Extracted text from the HTML intro
            - text (list): Parsed sections (format depends on 'structured' parameter)
            - surtitle (str): Always set to "Travail-Emploi"
            - source (str): Always set to "travail-emploi"
    """
    with open(os.path.join(target_dir)) as f:
        data = json.load(f)

    if structured:

        def join_sections(sections):
            texts = []
            for section in sections:
                texts.append(
                    {
                        "text": normalize(section["text"]),
                        "context": [normalize(section["title"])],
                    }
                )
            return texts

    else:

        def join_sections(sections):
            text = ""
            for section in sections:
                text += normalize(f"{section['title']}\n\n{section['text']}")

            return [text]

    docs = []
    for doc in data:
        sheet = {
            "title": normalize(doc["title"]),
            "url": doc["url"],
            "date": doc["date"],
            "sid": doc["pubId"],
            "introduction": get_text(BeautifulSoup(doc["intro"], "html.parser")),
            "text": join_sections(doc["sections"]),
            "surtitle": "Travail-Emploi",
            "source": "travail-emploi",
        }

        docs.append(sheet)

    return docs


class RagSource:
    # At this point a sheet is an hybrid dict data structure with with only a set of mandatory fields:
    # - "sid" -> unique identifier
    # - "title -> sheet title
    # - "text" -> main payload
    # - "context" -> successive subtitle (if structured=True)
    # - "source" -> The source of the sheet (service-public, vie-publique, legifrance, etc)
    # - "url" -> URL of the source
    # Depending on the source, they can have many more attribute...

    @classmethod
    def is_valid(cls, source):
        return source in cls.__dict__.values()

    @classmethod
    def get_sheets(
        cls, storage_dir: str | None, structured: bool = False
    ) -> list[dict]:
        """
        Retrieve and parse sheet data from a specified storage directory.

        This class method reads and parses sheet files from different data sources based on the
        storage directory path. It supports multiple data sources including Service Public
        (particulier and pro) and Travail Emploi.

        Args:
            storage_dir (str | None): The base path where files are stored and will be read from.
                Must be a valid directory path corresponding to one of the supported data sources.
            structured (bool, optional): Whether to parse the sheets in a structured format.
                Defaults to False.

        Returns:
            list: A list of dictionaries containing parsed sheet data. Each sheet dictionary
                includes at minimum a 'sid' (sheet ID) field. Duplicates based on 'sid' are
                automatically removed.

        Raises:
            ValueError: If storage_dir is None or empty.
            FileNotFoundError: If the expected directory structure for Travail Emploi data
                cannot be found.
            NotImplementedError: If the storage_dir does not match any known data source.

        Notes:
            - For Service Public data sources, files are read from a subdirectory matching
            the basename of storage_dir.
            - For Travail Emploi data, the method searches for a directory starting with
            the basename of storage_dir.
            - Duplicate sheets (identified by 'sid') are automatically removed, with logging
            of the duplicates found.
        """
        if not storage_dir:
            raise ValueError("You must give a storage directory.")

        sheets = []
        if SERVICE_PUBLIC_PART_DATA_FOLDER.endswith(
            storage_dir
        ) or SERVICE_PUBLIC_PRO_DATA_FOLDER.endswith(storage_dir):
            # storage_dir: the base path where files are gonna be written.
            # target_dir: read-only base path where sheets are read.
            target_dir = f"{storage_dir}/{storage_dir.split('/')[-1]}"
            sheets.extend(
                _parse_xml(
                    target_dir=target_dir, parse_type="text", structured=structured
                )
            )
        elif TRAVAIL_EMPLOI_DATA_FOLDER.endswith(storage_dir):
            base_name = os.path.basename(storage_dir)
            target_dir = next(
                (
                    os.path.join(storage_dir, f)
                    for f in os.listdir(storage_dir)
                    if f.startswith(base_name)
                ),
                None,
            )
            if target_dir is None:
                logger.error(
                    f"No file starting with '{base_name}' found in {storage_dir}"
                )
                raise FileNotFoundError()
            sheets.extend(_parse_travailEmploi(target_dir, structured=structured))
        else:
            raise NotImplementedError("Rag source unknown")

        # Remove duplicate
        sids = [x["sid"] for x in sheets]
        seen = set()
        to_remove = [i for i, sid in enumerate(sids) if sid in seen or seen.add(sid)]
        n_dup = len(to_remove)
        if n_dup > 0:
            logger.info(f"Dropping {n_dup} duplicated sheets")
            logger.info([sheets[i]["sid"] for i in to_remove])
        for ix in sorted(to_remove, reverse=True):
            sheets.pop(ix)

        return sheets
