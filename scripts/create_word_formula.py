from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W)
ET.register_namespace("m", M)


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def wrun(text: str, bold: bool = False) -> ET.Element:
    run = ET.Element(qn(W, "r"))
    if bold:
        props = ET.SubElement(run, qn(W, "rPr"))
        ET.SubElement(props, qn(W, "b"))
    node = ET.SubElement(run, qn(W, "t"))
    node.set(f"{{{XML}}}space", "preserve")
    node.text = text
    return run


def para(text: str, bold: bool = False) -> ET.Element:
    p = ET.Element(qn(W, "p"))
    p.append(wrun(text, bold))
    return p


def math_run(text: str) -> ET.Element:
    run = ET.Element(qn(M, "r"))
    node = ET.SubElement(run, qn(M, "t"))
    node.text = text
    return run


def subscript(base: str, lower: str) -> ET.Element:
    node = ET.Element(qn(M, "sSub"))
    e = ET.SubElement(node, qn(M, "e")); e.append(math_run(base))
    sub = ET.SubElement(node, qn(M, "sub")); sub.append(math_run(lower))
    return node


def sub_sup(base: str, lower: str, upper: str) -> ET.Element:
    node = ET.Element(qn(M, "sSubSup"))
    e = ET.SubElement(node, qn(M, "e")); e.append(math_run(base))
    sub = ET.SubElement(node, qn(M, "sub")); sub.append(math_run(lower))
    sup = ET.SubElement(node, qn(M, "sup")); sup.append(math_run(upper))
    return node


def fraction(numerator: list[ET.Element], denominator: list[ET.Element]) -> ET.Element:
    node = ET.Element(qn(M, "f"))
    num = ET.SubElement(node, qn(M, "num"))
    for item in numerator: num.append(item)
    den = ET.SubElement(node, qn(M, "den"))
    for item in denominator: den.append(item)
    return node


def math_cell(items: list[ET.Element]) -> ET.Element:
    cell = ET.Element(qn(M, "e"))
    for item in items: cell.append(item)
    return cell


def formula_paragraph() -> ET.Element:
    p = ET.Element(qn(W, "p"))
    ppr = ET.SubElement(p, qn(W, "pPr"))
    jc = ET.SubElement(ppr, qn(W, "jc")); jc.set(qn(W, "val"), "center")
    omath_para = ET.SubElement(p, qn(M, "oMathPara"))
    omath = ET.SubElement(omath_para, qn(M, "oMath"))
    omath.append(sub_sup("q", "k,i,c", "(0)"))
    omath.append(math_run(" = "))
    delimiter = ET.SubElement(omath, qn(M, "d"))
    dpr = ET.SubElement(delimiter, qn(M, "dPr"))
    beg = ET.SubElement(dpr, qn(M, "begChr")); beg.set(qn(M, "val"), "{")
    end = ET.SubElement(dpr, qn(M, "endChr")); end.set(qn(M, "val"), "")
    content = ET.SubElement(delimiter, qn(M, "e"))
    matrix = ET.SubElement(content, qn(M, "m"))
    first_row = ET.SubElement(matrix, qn(M, "mr"))
    first_row.append(math_cell([fraction([math_run("1")], [math_run("|"), subscript("Y", "k,i"), math_run("|")])]))
    first_row.append(math_cell([math_run("c ∈ "), subscript("Y", "k,i")]))
    second_row = ET.SubElement(matrix, qn(M, "mr"))
    second_row.append(math_cell([math_run("0")]))
    second_row.append(math_cell([math_run("c ∉ "), subscript("Y", "k,i")]))
    return p


def create(source: Path, output: Path) -> None:
    with zipfile.ZipFile(source, "r") as original:
        root = ET.fromstring(original.read("word/document.xml"))
        body = root.find(qn(W, "body"))
        if body is None:
            raise RuntimeError("No Word document body")
        sect_pr = body.find(qn(W, "sectPr"))
        if sect_pr is None:
            raise RuntimeError("No Word section settings")
        sect_pr = copy.deepcopy(sect_pr)
        for child in list(body):
            body.remove(child)
        body.append(para("标签置信度向量初始化公式（Word 原生公式对象）", bold=True))
        body.append(para("可直接在 Word 中选中下列公式并复制至目标文档。"))
        body.append(formula_paragraph())
        body.append(para("其中，c∈Y_(k,i) 时取第一行；c∉Y_(k,i) 时取第二行。"))
        body.append(sect_pr)
        document = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as result:
            for item in original.infolist():
                payload = document if item.filename == "word/document.xml" else original.read(item.filename)
                result.writestr(item, payload)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: create_word_formula.py SOURCE.docx OUTPUT.docx")
    create(Path(sys.argv[1]), Path(sys.argv[2]))
