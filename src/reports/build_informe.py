"""Genera reports/informe/Informe_Proyecto_Final.docx desde reports/Trabajo_Escrito.md.

Uso: /private/tmp/claude-501/venv/bin/python src/reports/build_informe.py [--pdf]
- DOCX: parser markdown propio + python-docx (caratula, estilos, tablas, imagenes, codigo).
- PDF: se exporta el DOCX con Microsoft Word via AppleScript (flag --pdf); fallback: ver
  nota en el log si Word falla (HTML + Chrome headless --print-to-pdf).
"""
import re, subprocess, sys
from pathlib import Path
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "reports/Trabajo_Escrito.md"
OUT = ROOT / "reports/informe/Informe_Proyecto_Final.docx"
LOGO = ROOT / "reports/informe/logo_uvg.png"
FONT = "Calibri"


def shade(el_pr, fill):
    s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:fill"), fill)
    el_pr.append(s)


def set_font(run, name=FONT, size=None, bold=None, italic=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*|\[[^\]]+\]\([^)]+\))")


def add_inline(p, text, size=None):
    for part in INLINE.split(text):
        if not part: continue
        if part.startswith("**"): r = p.add_run(part[2:-2]); set_font(r, size=size, bold=True)
        elif part.startswith("`"): r = p.add_run(part[1:-1]); set_font(r, "Consolas", (size or 11) - 1)
        elif part.startswith("*") and len(part) > 2: r = p.add_run(part[1:-1]); set_font(r, size=size, italic=True)
        elif part.startswith("["):
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part)
            r = p.add_run(f"{m[1]} ({m[2]})"); set_font(r, size=size)
        else: r = p.add_run(part); set_font(r, size=size)


def center(doc, text="", size=11, bold=False, after=0):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(after)
    if text: set_font(p.add_run(text), size=size, bold=bold)
    return p


def cover(doc):
    center(doc, "UNIVERSIDAD DEL VALLE DE GUATEMALA", bold=True)
    for t in ("CC3092 - DEEP LEARNING Y SISTEMAS INTELIGENTES", "SECCIÓN – 20", "Ing. Javier Fong"):
        center(doc, t)
    center(doc, after=14)
    p = center(doc, after=14); p.add_run().add_picture(str(LOGO), width=Cm(5.5))
    center(doc, "Proyecto Final. Recomendador de moda para e-commerce en Guatemala", size=26, after=30)
    for a in ("Javier André Benítez García 23405", "Javier Alejandro López del Cid 23415",
              "Hugo Javier Cifuentes Carredano 23079"):
        center(doc, a)
    for _ in range(5): center(doc)
    center(doc, "GUATEMALA, 18 de noviembre de 2026", bold=True)


def page_number_footer(section):
    section.different_first_page_header_footer = True  # caratula sin numero
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(); set_font(r, size=10)
    for typ, txt in (("begin", None), (None, " PAGE "), ("end", None)):
        if typ:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), typ)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._element.append(e)


def table(doc, rows):
    n = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=n); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        for k, v in (("val", "single"), ("sz", "4"), ("color", "A6A6A6")): e.set(qn(f"w:{k}"), v)
        b.append(e)
    tblPr.append(b)
    w = OxmlElement("w:tblW"); w.set(qn("w:w"), "5000"); w.set(qn("w:type"), "pct")
    for old in tblPr.findall(qn("w:tblW")): tblPr.remove(old)
    tblPr.append(w)
    for i, row in enumerate(rows):
        for j in range(n):
            c = t.cell(i, j); p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            txt = row[j] if j < len(row) else ""
            add_inline(p, f"**{txt}**" if i == 0 and not txt.startswith("**") and txt else txt, size=9)
            if i == 0: shade(c._tc.get_or_add_tcPr(), "D9E2F3")
    if rows:  # repetir encabezado
        h = OxmlElement("w:tblHeader"); t.rows[0]._tr.get_or_add_trPr().append(h)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def code_block(doc, lines):
    for ln in lines:
        p = doc.add_paragraph(); pf = p.paragraph_format
        pf.space_after = Pt(0); pf.left_indent = Cm(0.2)
        shade(p._p.get_or_add_pPr(), "F2F2F2")
        set_font(p.add_run(ln or " "), "Consolas", 9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def image(doc, alt, rel):
    path = MD.parent / rel
    w, h = Image.open(path).size
    width = min(15.0, 22.0 * w / h)  # limita ancho 15 cm y alto ~22 cm
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width))
    if alt:
        c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(c.add_run(alt), size=9, italic=True)


def setup_styles(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.59), Cm(27.94)
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.5)
    n = doc.styles["Normal"]; n.font.name = FONT; n.font.size = Pt(11)
    n.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    n.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
        st = doc.styles[name]; st.font.name = FONT; st.font.size = Pt(size)
        st.font.bold = True; st.font.color.rgb = RGBColor(0, 0, 0)
        rf = st.element.rPr.rFonts
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme"):
            if rf.get(qn(a)): del rf.attrib[qn(a)]
        st.paragraph_format.space_before = Pt(14); st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True


def build():
    doc = Document(); setup_styles(doc)
    cover(doc)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    page_number_footer(doc.sections[0])
    lines = MD.read_text(encoding="utf-8").split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("## Resumen")][0]
    para = []
    def flush():
        if para: add_inline(doc.add_paragraph(), " ".join(para)); para.clear()
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            flush(); j = i + 1; buf = []
            while not lines[j].startswith("```"): buf.append(lines[j]); j += 1
            code_block(doc, buf); i = j + 1; continue
        m = re.match(r"^(#{2,4}) (.*)", l)
        if m:
            flush(); doc.add_heading(m[2].strip(), level=len(m[1]) - 1); i += 1; continue
        if l.strip() == "---" or not l.strip(): flush(); i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", l)
        if m: flush(); image(doc, m[1], m[2]); i += 1; continue
        if l.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells): rows.append(cells)
                i += 1
            table(doc, rows); continue
        m = re.match(r"^(\s*)([-*]|\d+\.) (.*)", l)
        if m:
            flush()
            p = doc.add_paragraph(style="List Number" if m[2][0].isdigit() else "List Bullet")
            p.paragraph_format.space_after = Pt(3); add_inline(p, m[3]); i += 1; continue
        para.append(l.strip()); i += 1
    flush()
    doc.save(OUT); print("OK", OUT)


def pdf_via_word():
    pdf = OUT.with_suffix(".pdf")
    script = f'''tell application "Microsoft Word"
 open POSIX file "{OUT}"
 set d to active document
 save as d file name "{pdf}" file format format PDF
 close d saving no
end tell'''
    subprocess.run(["osascript", "-e", script], check=True, timeout=240)
    print("OK", pdf)


def pdf_via_chrome():
    """Fallback usado (Word se colgo al exportar): HTML equivalente + Chrome headless."""
    import html as H
    def inl(t):
        t = H.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*", r"<i>\1</i>", t)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", t)
    lines = MD.read_text(encoding="utf-8").split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("## Resumen")][0]
    o, para, lst = [], [], None
    def flush():
        nonlocal lst
        if para: o.append("<p>" + inl(" ".join(para)) + "</p>"); para.clear()
        if lst: o.append(f"</{lst}>"); lst = None
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            flush(); j = i + 1; buf = []
            while not lines[j].startswith("```"): buf.append(lines[j]); j += 1
            o.append("<pre>" + H.escape("\n".join(buf)) + "</pre>"); i = j + 1; continue
        m = re.match(r"^(#{2,4}) (.*)", l)
        if m: flush(); o.append(f"<h{len(m[1])-1}>{inl(m[2])}</h{len(m[1])-1}>"); i += 1; continue
        if l.strip() == "---" or not l.strip(): flush(); i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", l)
        if m:
            flush(); o.append(f'<figure><img src="{(MD.parent / m[2]).as_uri()}"><figcaption>{H.escape(m[1])}</figcaption></figure>'); i += 1; continue
        if l.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].startswith("|"):
                c = [x.strip().replace("\\|", "|") for x in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))]
                if not all(re.fullmatch(r":?-+:?", x) for x in c): rows.append(c)
                i += 1
            o.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if r is rows[0] else 'td'}>{inl(x)}</{'th' if r is rows[0] else 'td'}>" for x in r) + "</tr>" for r in rows) + "</table>"); continue
        m = re.match(r"^(\s*)([-*]|\d+\.) (.*)", l)
        if m:
            tag = "ol" if m[2][0].isdigit() else "ul"
            if lst != tag: flush(); o.append(f"<{tag}>"); lst = tag
            o.append(f"<li>{inl(m[3])}</li>"); i += 1; continue
        para.append(l.strip()); i += 1
    flush()
    css = """@page{size:Letter;margin:2.5cm}body{font:11pt Calibri,Carlito,sans-serif}
.cov{height:22.2cm;text-align:center;page-break-after:always;position:relative}.cov p{margin:0}
.cov img{width:5.5cm;margin:26px 0}.cov h1{font:400 26pt Calibri,sans-serif;margin:0 0 40px}
.cov .d{position:absolute;bottom:0;width:100%;font-weight:700}
h1{font-size:16pt;margin:18px 0 6px;break-after:avoid}h2{font-size:13pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}
table{width:100%;border-collapse:collapse;font-size:9pt;margin:8px 0}th,td{border:.5pt solid #a6a6a6;padding:2px 4px;vertical-align:top}
th{background:#d9e2f3}tr{break-inside:avoid}pre{font:9pt Consolas,Menlo,monospace;background:#f2f2f2;padding:6px;white-space:pre-wrap}
code{font:10pt Consolas,Menlo,monospace}figure{margin:10px 0;text-align:center;break-inside:avoid}
figure img{max-width:15cm;max-height:20cm}figcaption{font-size:9pt;font-style:italic}li{margin:2px 0}"""
    cov = f"""<div class=cov><p><b>UNIVERSIDAD DEL VALLE DE GUATEMALA</b></p><p>CC3092 - DEEP LEARNING Y SISTEMAS INTELIGENTES</p>
<p>SECCIÓN – 20</p><p>Ing. Javier Fong</p><img src="{LOGO.as_uri()}"><h1>Proyecto Final. Recomendador de moda para e-commerce en Guatemala</h1>
<p>Javier André Benítez García 23405</p><p>Javier Alejandro López del Cid 23415</p><p>Hugo Javier Cifuentes Carredano 23079</p>
<p class=d>GUATEMALA, 18 de noviembre de 2026</p></div>"""
    f = OUT.with_suffix(".html")
    f.write_text(f'<meta charset="utf-8"><style>{css}</style>{cov}' + "\n".join(o), encoding="utf-8")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # numero de pagina: Chrome no lo da sin header/footer; se estampa luego (ver stamp_numbers)
    subprocess.run([chrome, "--headless", "--no-pdf-header-footer", f"--print-to-pdf={OUT.with_suffix('.pdf')}", f.as_uri()], check=True, timeout=180)
    f.unlink()
    stamp_numbers(OUT.with_suffix(".pdf")); print("OK pdf")


def stamp_numbers(pdf):
    """Numera paginas desde la 2 (la caratula queda sin numero)."""
    import io
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    w = PdfWriter(clone_from=str(pdf))
    for n, pg in enumerate(w.pages, 1):
        if n > 1:
            b = io.BytesIO(); W, Hh = float(pg.mediabox.width), float(pg.mediabox.height)
            c = canvas.Canvas(b, pagesize=(W, Hh)); c.setFont("Helvetica", 10)
            c.drawCentredString(W / 2, 40, str(n)); c.save(); b.seek(0)
            pg.merge_page(PdfReader(b).pages[0])
    w.write(str(pdf))


if __name__ == "__main__":
    build()
    if "--pdf" in sys.argv: pdf_via_word()
    if "--chrome" in sys.argv: pdf_via_chrome()
