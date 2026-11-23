"""用 Word COM 把 docx 导出为 PDF，再用 PyMuPDF 渲染为 PNG，便于检查排版。

用法：python tools/docx_preview.py report/项目设计文档.docx
输出：outputs/_preview/page-XX.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "report" / "项目设计文档.docx")
    src = src if src.is_absolute() else (ROOT / src)
    pdf = src.with_suffix(".pdf")

    import win32com.client as win32
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(src), ReadOnly=True)
        # 更新所有域（页码等）
        try:
            doc.Fields.Update()
        except Exception:
            pass
        doc.SaveAs(str(pdf), FileFormat=17)  # wdFormatPDF
        doc.Close(False)
    finally:
        word.Quit()

    import fitz
    out = ROOT / "outputs" / "_preview"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("page-*.png"):
        f.unlink()
    with fitz.open(pdf) as d:
        print(f"[PDF] {pdf}  共 {d.page_count} 页")
        for i, page in enumerate(d):
            pix = page.get_pixmap(dpi=110)
            pix.save(out / f"page-{i + 1:02d}.png")
    print(f"[PNG] {out}")


if __name__ == "__main__":
    main()
