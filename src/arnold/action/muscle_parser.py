"""
Парсер мышц из MuJoCo XML для модели MyoHuman.

Извлекает список actuator names из XML-файлов.
Порядок мышц соответствует порядку в финальной модели MuJoCo
(torso → legs → arms).
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

# Порядок asset-файлов в myohuman.xml (определяет порядок actuators)
MYOHUMAN_ASSET_ORDER = [
    "assets/myotorso_assets.xml",
    "assets/myolegs_assets.xml",
    "assets/myoarm_assets.xml",
]


def _parse_actuators_from_xml(xml_path: Path) -> List[str]:
    """
    Извлекает имена actuators из одного XML-файла.
    Ищет <actuator> и внутри — <general name="...">, <muscle name="...">.
    """
    actuators: List[str] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"XML parse error in {xml_path}: {e}")
        return actuators

    for actuator_elem in root.iter():
        tag = actuator_elem.tag.split("}")[-1] if "}" in actuator_elem.tag else actuator_elem.tag
        if tag != "actuator":
            continue
        for child in actuator_elem:
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            name = child.get("name")
            if name and ctag in ("general", "muscle", "motor"):
                actuators.append(name)

    return actuators


def parse_muscles_from_xml(
    xml_path: str | Path,
    asset_dirs: Optional[List[str | Path]] = None,
) -> List[str]:
    """
    Парсит все мышцы из MuJoCo XML MyoHuman.

    Использует фиксированный порядок asset-файлов для корректного
    соответствия порядку actuators в загруженной модели.

    Args:
        xml_path: Путь к главному XML или к директории xml
        asset_dirs: Дополнительные директории для поиска asset-файлов

    Returns:
        Список имён actuators в порядке их появления в модели.
    """
    xml_path = Path(xml_path).resolve()
    if xml_path.is_dir():
        base_dir = xml_path
    else:
        base_dir = xml_path.parent

    if asset_dirs is None:
        search_dirs = [base_dir]
    else:
        search_dirs = [Path(d).resolve() for d in asset_dirs]
        if base_dir not in search_dirs:
            search_dirs.insert(0, base_dir)

    all_actuators: List[str] = []
    for rel_path in MYOHUMAN_ASSET_ORDER:
        for d in search_dirs:
            p = d / rel_path
            if p.exists():
                acts = _parse_actuators_from_xml(p)
                all_actuators.extend(acts)
                logger.debug(f"From {rel_path}: {len(acts)} actuators")
                break
        else:
            logger.warning(f"Asset file not found: {rel_path}")

    logger.info(f"Parsed {len(all_actuators)} muscles total")
    return all_actuators


def parse_muscles_myohuman(xml_root: Optional[Path] = None) -> List[str]:
    """
    Парсит мышцы для модели MyoHuman по умолчанию.

    Args:
        xml_root: Корень XML (по умолчанию experts/Myohuman/xml)

    Returns:
        Список имён мышц в порядке actuators модели.
    """
    if xml_root is None:
        # arnold/action/muscle_parser.py -> arnold/experts/Myohuman/xml
        xml_root = Path(__file__).resolve().parent.parent / "experts" / "Myohuman" / "xml"
    xml_path = xml_root / "myohuman.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"MyoHuman XML not found: {xml_path}")

    return parse_muscles_from_xml(xml_path, asset_dirs=[xml_root])
