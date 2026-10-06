import os
import io
import time
import uuid
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# Standard disclaimers per Phase 6 specifications
DISCLAIMER_GENERATED = (
    "COMPUTER-GENERATED APPEARANCE: These images are computer-generated appearance variations. "
    "This image is a visual hypothesis and is not a confirmed photograph of the person."
)
DISCLAIMER_PROBABLE = (
    "COMPUTER-GENERATED PROBABLE APPEARANCE: Computer-generated probable appearance. "
    "This image is a visual hypothesis and is not a confirmed current photograph."
)


def get_optimized_image_reader(img_path, max_dim=800):
    """
    Loads an image with Pillow, ensures RGB on white background,
    and returns a compressed in-memory JPEG ImageReader with dimensions.
    """
    if not img_path or not os.path.exists(img_path):
        return None, 100, 100

    try:
        with Image.open(img_path) as pil_img:
            if pil_img.mode in ('RGBA', 'LA'):
                bg = Image.new('RGB', pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = bg
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            orig_w, orig_h = pil_img.size
            if max(orig_w, orig_h) > max_dim:
                scale = max_dim / float(max(orig_w, orig_h))
                new_size = (int(orig_w * scale), int(orig_h * scale))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=92, optimize=True)
            buf.seek(0)
            return ImageReader(buf), orig_w, orig_h
    except Exception:
        return None, 100, 100


def draw_fitted_image(c, img_path, x, y, max_w, max_h, label=None, label_bg=colors.HexColor("#0f172a"), border_color=colors.HexColor("#334155"), border_width=1.5):
    """
    Draws an image fitted inside (x, y, max_w, max_h) preserving aspect ratio.
    Optional label banner under or over the image.
    """
    reader, orig_w, orig_h = get_optimized_image_reader(img_path)

    if reader is None:
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.rect(x, y, max_w, max_h, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x + max_w / 2, y + max_h / 2, "[ Image Unavailable ]")
        return

    aspect = orig_w / float(orig_h)
    target_aspect = max_w / float(max_h)

    if aspect > target_aspect:
        draw_w = max_w
        draw_h = max_w / aspect
        draw_x = x
        draw_y = y + (max_h - draw_h) / 2
    else:
        draw_h = max_h
        draw_w = max_h * aspect
        draw_x = x + (max_w - draw_w) / 2
        draw_y = y

    # Background card
    c.setFillColor(colors.white)
    c.setStrokeColor(border_color)
    c.setLineWidth(border_width)
    c.rect(draw_x, draw_y, draw_w, draw_h, fill=1, stroke=1)

    # Draw image
    c.drawImage(reader, draw_x, draw_y, width=draw_w, height=draw_h, preserveAspectRatio=True)

    # Optional label strip at bottom of image
    if label:
        lbl_h = 5.5 * mm
        c.setFillColor(label_bg)
        c.rect(draw_x, draw_y, draw_w, lbl_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(draw_x + draw_w / 2, draw_y + 1.8 * mm, label.upper())


def draw_contact_box(c, x, y, w, h, phone, alt_phone=None, contact_person=None, case_no=None, agency=None, style="standard"):
    """
    Draws a high-visibility wall-poster contact callout block with prominent phone numbers.
    """
    if style == "urgent" or style == "crimson":
        bg_color = colors.HexColor("#7f1d1d")       # Deep Crimson
        border_color = colors.HexColor("#dc2626")   # Red
        title_color = colors.HexColor("#fef08a")    # Yellow/Gold
        phone_color = colors.white
    elif style == "gold":
        bg_color = colors.HexColor("#78350f")       # Amber-900
        border_color = colors.HexColor("#f59e0b")   # Amber-500
        title_color = colors.HexColor("#fef3c7")
        phone_color = colors.white
    else:
        bg_color = colors.HexColor("#0f172a")       # Slate-900
        border_color = colors.HexColor("#0284c7")   # Sky-600
        title_color = colors.HexColor("#38bdf8")    # Sky-400
        phone_color = colors.white

    c.setFillColor(bg_color)
    c.setStrokeColor(border_color)
    c.setLineWidth(2)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)

    # Top header inside contact box
    c.setFillColor(title_color)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawCentredString(x + w / 2, y + h - 6 * mm, "IF YOU HAVE ANY INFORMATION, PLEASE CONTACT IMMEDIATELY:")

    # Giant prominent phone number
    c.setFillColor(phone_color)
    c.setFont("Helvetica-Bold", 16)
    phone_str = f"PHONE: {phone or 'NOT PROVIDED'}"
    if alt_phone:
        phone_str += f"   |   ALT: {alt_phone}"
    c.drawCentredString(x + w / 2, y + h - 13.5 * mm, phone_str)

    # Sub-details: Contact Person, Case Number, Agency
    sub_items = []
    if contact_person:
        sub_items.append(f"CONTACT: {contact_person}")
    if case_no:
        sub_items.append(f"CASE REF: {case_no}")
    if agency:
        sub_items.append(f"AGENCY: {agency}")

    if sub_items:
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(x + w / 2, y + 3 * mm, "   •   ".join(sub_items))


def draw_disclaimer_footer(c, page_w, y, max_w, text_type="standard"):
    """
    Renders the legal disclaimer footer.
    """
    if text_type == "probable":
        text = DISCLAIMER_PROBABLE
    else:
        text = DISCLAIMER_GENERATED

    x = (page_w - max_w) / 2
    h = 10 * mm

    c.setFillColor(colors.HexColor("#f1f5f9"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(0.8)
    c.roundRect(x, y, max_w, h, 3, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica-Oblique", 6.8)
    # Wrap text cleanly in 2 lines
    parts = text.split(":")
    tag = parts[0] + ":"
    body = parts[1].strip() if len(parts) > 1 else text
    c.drawCentredString(page_w / 2, y + 5.5 * mm, tag)
    c.drawCentredString(page_w / 2, y + 2.2 * mm, body)


# =========================================================================
# TEMPLATE 1: PERSON INFORMATION POSTER
# Purpose: General person-information display flyer for wall posting.
# =========================================================================
def render_template_1(c, page_w, page_h, primary_face, addl_faces, data, meta):
    margin_x = 15 * mm
    avail_w = page_w - 2 * margin_x
    top_y = page_h - 15 * mm

    # Top Header Banner
    title = (data.get('custom_title') or 'PERSON INFORMATION').upper()
    c.setFillColor(colors.HexColor("#0f172a"))  # Slate-900
    c.rect(margin_x, top_y - 18 * mm, avail_w, 18 * mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_w / 2, top_y - 12.5 * mm, title)

    # Sub-header reference line
    c.setFillColor(colors.HexColor("#e2e8f0"))
    c.setFont("Helvetica-Bold", 8)
    ref_text = f"CASE REF: {data.get('case_number') or meta.get('ref_no') or 'TE-2026-001'}   •   DATE: {data.get('date_missing') or meta.get('date') or time.strftime('%d/%m/%Y')}"
    c.drawCentredString(page_w / 2, top_y - 16.5 * mm, ref_text)

    # Large Center Photo Area
    photo_box_top = top_y - 23 * mm
    photo_h = 105 * mm
    photo_w = 95 * mm
    photo_x = (page_w - photo_w) / 2
    photo_y = photo_box_top - photo_h

    lbl = primary_face.get('label') or ('LAST KNOWN PHOTOGRAPH' if primary_face.get('source_type') == 'real_photo' else 'COMPUTER-GENERATED RECONSTRUCTION')
    draw_fitted_image(c, primary_face.get('image_path', ''), photo_x, photo_y, photo_w, photo_h, label=lbl, border_color=colors.HexColor("#0f172a"), border_width=2)

    # Large Prominent Name Banner
    name_y = photo_y - 14 * mm
    name_str = (data.get('full_name') or 'SUBJECT NAME NOT SPECIFIED').upper()
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(page_w / 2, name_y, name_str)

    # Physical Attributes Quick Badges Grid
    attr_box_y = name_y - 28 * mm
    attr_box_h = 24 * mm
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, attr_box_y, avail_w, attr_box_h, 4, fill=1, stroke=1)

    attrs = [
        ("AGE", data.get('age', 'Unknown')),
        ("GENDER", data.get('gender', 'Unknown')),
        ("HEIGHT", data.get('height', 'Unknown')),
        ("BUILD", data.get('build', 'Unknown')),
        ("HAIR", data.get('hair', 'Unknown')),
        ("EYES", data.get('eyes', 'Unknown'))
    ]

    col_w = avail_w / 6.0
    for idx, (k, v) in enumerate(attrs):
        cx = margin_x + idx * col_w
        if idx > 0:
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.line(cx, attr_box_y + 2 * mm, cx, attr_box_y + attr_box_h - 2 * mm)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 7.5)
        c.drawCentredString(cx + col_w / 2, attr_box_y + attr_box_h - 7 * mm, k)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(cx + col_w / 2, attr_box_y + 6 * mm, str(v)[:14])

    # Description & Details Section
    desc_y = attr_box_y - 26 * mm
    desc_h = 22 * mm
    c.setFillColor(colors.HexColor("#ffffff"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, desc_y, avail_w, desc_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#0284c7"))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin_x + 5 * mm, desc_y + desc_h - 6 * mm, "PHYSICAL DESCRIPTION & IDENTIFYING INFORMATION:")

    desc_text = data.get('description') or data.get('notes') or "No additional physical details provided."
    marks = data.get('identifying_marks')
    if marks:
        desc_text = f"Identifying Marks: {marks}. " + desc_text

    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica", 8.5)
    # Wrap in 2 lines
    c.drawString(margin_x + 5 * mm, desc_y + desc_h - 12 * mm, desc_text[:110])
    if len(desc_text) > 110:
        c.drawString(margin_x + 5 * mm, desc_y + desc_h - 17 * mm, desc_text[110:220])

    # Prominent Contact Box
    contact_h = 24 * mm
    contact_y = desc_y - contact_h - 5 * mm
    draw_contact_box(c, margin_x, contact_y, avail_w, contact_h,
                     phone=data.get('phone', ''),
                     alt_phone=data.get('alt_phone', ''),
                     contact_person=data.get('contact_person', ''),
                     case_no=data.get('case_number', ''),
                     agency=data.get('police_station', ''),
                     style="standard")

    # Disclaimer at bottom if non-real photo used
    has_generated = primary_face.get('source_type') != 'real_photo'
    if has_generated:
        draw_disclaimer_footer(c, page_w, 8 * mm, avail_w, text_type="standard")


# =========================================================================
# TEMPLATE 2: IMAGE + MULTIPLE APPEARANCES POSTER
# Purpose: Primary face + 1 to 4 generated appearance variations.
# =========================================================================
def render_template_2(c, page_w, page_h, primary_face, addl_faces, data, meta):
    margin_x = 15 * mm
    avail_w = page_w - 2 * margin_x
    top_y = page_h - 15 * mm

    # Header
    title = (data.get('custom_title') or 'PERSON INFORMATION & APPEARANCE VARIATIONS').upper()
    c.setFillColor(colors.HexColor("#1e293b"))
    c.rect(margin_x, top_y - 16 * mm, avail_w, 16 * mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 17)
    c.drawCentredString(page_w / 2, top_y - 11 * mm, title)

    # Top Area: Primary Face (Left) + Possible Appearances Grid (Right)
    top_area_y = top_y - 20 * mm
    visual_h = 92 * mm
    primary_w = 75 * mm
    primary_x = margin_x
    primary_y = top_area_y - visual_h

    # Primary Image on Left
    p_lbl = primary_face.get('label') or ('LAST KNOWN PHOTOGRAPH' if primary_face.get('source_type') == 'real_photo' else 'BASE RECONSTRUCTION')
    draw_fitted_image(c, primary_face.get('image_path', ''), primary_x, primary_y, primary_w, visual_h, label=p_lbl, border_color=colors.HexColor("#0284c7"), border_width=2)

    # Right Area: Possible Appearances
    right_x = primary_x + primary_w + 6 * mm
    right_w = avail_w - primary_w - 6 * mm

    # Section title
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x, top_area_y - 4 * mm, "POSSIBLE APPEARANCES / HYPOTHESES:")

    sub_faces = addl_faces[:4]
    n_sub = len(sub_faces)

    if n_sub > 0:
        sub_grid_top = top_area_y - 7 * mm
        sub_grid_h = visual_h - 7 * mm
        # 2x2 grid if 3 or 4, or 1 row if 1 or 2
        rows = 2 if n_sub > 2 else 1
        cols = 2 if n_sub > 1 else 1
        cell_w = (right_w - (cols - 1) * 4 * mm) / cols
        cell_h = (sub_grid_h - (rows - 1) * 4 * mm) / rows

        for idx, s_face in enumerate(sub_faces):
            r = idx // cols
            col = idx % cols
            cx = right_x + col * (cell_w + 4 * mm)
            cy = sub_grid_top - (r + 1) * cell_h - r * 4 * mm
            s_lbl = "COMPUTER-GENERATED APPEARANCE"
            draw_fitted_image(c, s_face.get('image_path', ''), cx, cy, cell_w, cell_h, label=s_lbl, label_bg=colors.HexColor("#334155"), border_width=1)
    else:
        # If no additional variations selected, show detailed case notes in right box
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.roundRect(right_x, primary_y, right_w, visual_h - 7 * mm, 4, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica", 9)
        c.drawString(right_x + 6 * mm, primary_y + visual_h - 20 * mm, "No additional appearance variations enrolled.")

    # Name Banner
    name_y = primary_y - 12 * mm
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_w / 2, name_y, (data.get('full_name') or 'SUBJECT FULL NAME').upper())

    # Biometric Attributes Table
    bio_y = name_y - 26 * mm
    bio_h = 22 * mm
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, bio_y, avail_w, bio_h, 4, fill=1, stroke=1)

    attrs = [
        ("AGE", data.get('age', 'Unknown')),
        ("GENDER", data.get('gender', 'Unknown')),
        ("HEIGHT", data.get('height', 'Unknown')),
        ("BUILD", data.get('build', 'Unknown')),
        ("HAIR", data.get('hair', 'Unknown')),
        ("EYES", data.get('eyes', 'Unknown'))
    ]
    col_w = avail_w / 6.0
    for idx, (k, v) in enumerate(attrs):
        cx = margin_x + idx * col_w
        if idx > 0:
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.line(cx, bio_y + 2 * mm, cx, bio_y + bio_h - 2 * mm)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(cx + col_w / 2, bio_y + bio_h - 6 * mm, k)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 9.5)
        c.drawCentredString(cx + col_w / 2, bio_y + 5 * mm, str(v)[:14])

    # Narrative / Description
    narr_y = bio_y - 24 * mm
    narr_h = 20 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, narr_y, avail_w, narr_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#0284c7"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(margin_x + 4 * mm, narr_y + narr_h - 5 * mm, "DETAILS & IDENTIFYING INFORMATION:")
    info_text = data.get('description') or data.get('notes') or "Investigative composite appearance hypotheses."
    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica", 8)
    c.drawString(margin_x + 4 * mm, narr_y + narr_h - 11 * mm, info_text[:115])
    if len(info_text) > 115:
        c.drawString(margin_x + 4 * mm, narr_y + narr_h - 16 * mm, info_text[115:230])

    # Contact Box
    contact_h = 24 * mm
    contact_y = narr_y - contact_h - 4 * mm
    draw_contact_box(c, margin_x, contact_y, avail_w, contact_h,
                     phone=data.get('phone', ''),
                     alt_phone=data.get('alt_phone', ''),
                     contact_person=data.get('contact_person', ''),
                     case_no=data.get('case_number', ''),
                     style="standard")

    # Disclaimer
    draw_disclaimer_footer(c, page_w, 8 * mm, avail_w, text_type="standard")


# =========================================================================
# TEMPLATE 3: PUBLIC INFORMATION / PERSON OF INTEREST
# Purpose: Authorized alert, identification notice, or public flyer.
# =========================================================================
def render_template_3(c, page_w, page_h, primary_face, addl_faces, data, meta):
    margin_x = 15 * mm
    avail_w = page_w - 2 * margin_x
    top_y = page_h - 15 * mm

    # Bold Alert Header Banner (Deep Navy or Crimson)
    title = (data.get('custom_title') or 'PERSON OF INTEREST').upper()
    c.setFillColor(colors.HexColor("#1e1b4b"))  # Dark Indigo / Navy
    c.rect(margin_x, top_y - 20 * mm, avail_w, 20 * mm, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#facc15"))  # Bright Yellow
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(page_w / 2, top_y - 11 * mm, title)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawCentredString(page_w / 2, top_y - 17 * mm, "PUBLIC IDENTIFICATION & INFORMATION BULLETIN")

    # Primary Image + Optional Secondary Appearances
    visual_top = top_y - 24 * mm
    visual_h = 100 * mm

    has_addl = len(addl_faces) > 0
    if has_addl:
        main_w = 80 * mm
        main_x = margin_x
        draw_fitted_image(c, primary_face.get('image_path', ''), main_x, visual_top - visual_h, main_w, visual_h,
                          label="PRIMARY RECORD / PHOTO", border_color=colors.HexColor("#1e1b4b"), border_width=2)

        # Right side: up to 2 or 3 additional variations
        side_x = main_x + main_w + 5 * mm
        side_w = avail_w - main_w - 5 * mm
        sub_list = addl_faces[:3]
        sub_h = (visual_h - (len(sub_list) - 1) * 3 * mm) / len(sub_list)

        for idx, sf in enumerate(sub_list):
            sy = visual_top - (idx + 1) * sub_h - idx * 3 * mm
            draw_fitted_image(c, sf.get('image_path', ''), side_x, sy, side_w, sub_h,
                              label=sf.get('label', 'Possible Appearance'), border_color=colors.HexColor("#64748b"), border_width=1)
    else:
        # Full width centered main image
        main_w = 95 * mm
        main_x = (page_w - main_w) / 2
        draw_fitted_image(c, primary_face.get('image_path', ''), main_x, visual_top - visual_h, main_w, visual_h,
                          label="SUBJECT RECORD PHOTO", border_color=colors.HexColor("#1e1b4b"), border_width=2)

    # Name Banner
    name_y = visual_top - visual_h - 12 * mm
    c.setFillColor(colors.HexColor("#1e1b4b"))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(page_w / 2, name_y, (data.get('full_name') or 'SUBJECT NAME / ALIAS').upper())

    # Case & Physical Table
    table_y = name_y - 30 * mm
    table_h = 26 * mm
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, table_y, avail_w, table_h, 4, fill=1, stroke=1)

    row1 = [
        ("CASE NUMBER", data.get('case_number', 'N/A')),
        ("AGE", data.get('age', 'Unknown')),
        ("GENDER", data.get('gender', 'Unknown')),
        ("HEIGHT", data.get('height', 'Unknown'))
    ]
    row2 = [
        ("BUILD", data.get('build', 'Unknown')),
        ("HAIR", data.get('hair', 'Unknown')),
        ("EYES", data.get('eyes', 'Unknown')),
        ("MARKS", data.get('identifying_marks', 'None Recorded'))
    ]

    col_w = avail_w / 4.0
    for idx, (k, v) in enumerate(row1):
        cx = margin_x + idx * col_w
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 7)
        c.drawString(cx + 4 * mm, table_y + table_h - 6 * mm, k + ":")
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(cx + 4 * mm, table_y + table_h - 11 * mm, str(v)[:18])

    for idx, (k, v) in enumerate(row2):
        cx = margin_x + idx * col_w
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 7)
        c.drawString(cx + 4 * mm, table_y + 9 * mm, k + ":")
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(cx + 4 * mm, table_y + 4 * mm, str(v)[:18])

    # Narrative Details Box
    desc_y = table_y - 24 * mm
    desc_h = 20 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, desc_y, avail_w, desc_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#1e1b4b"))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin_x + 4 * mm, desc_y + desc_h - 5 * mm, "SUMMARY OF INFORMATION & CIRCUMSTANCES:")

    summary = data.get('description') or data.get('notes') or "Law enforcement identification composite."
    c.setFillColor(colors.HexColor("#334155"))
    c.setFont("Helvetica", 8)
    c.drawString(margin_x + 4 * mm, desc_y + desc_h - 11 * mm, summary[:120])
    if len(summary) > 120:
        c.drawString(margin_x + 4 * mm, desc_y + desc_h - 16 * mm, summary[120:240])

    # High-Impact Contact / Tip Line Box
    contact_h = 24 * mm
    contact_y = desc_y - contact_h - 4 * mm
    draw_contact_box(c, margin_x, contact_y, avail_w, contact_h,
                     phone=data.get('phone', ''),
                     alt_phone=data.get('alt_phone', ''),
                     contact_person=data.get('contact_person', ''),
                     case_no=data.get('case_number', ''),
                     agency=data.get('police_station', ''),
                     style="standard")

    # Disclaimer if generated variations were used
    if primary_face.get('source_type') != 'real_photo' or has_addl:
        draw_disclaimer_footer(c, page_w, 8 * mm, avail_w, text_type="standard")


# =========================================================================
# TEMPLATE 4: MISSING PERSON — REAL PHOTOGRAPH
# Purpose: Primary missing-person flyer using real photograph. No sketch needed.
# =========================================================================
def render_template_4(c, page_w, page_h, primary_face, addl_faces, data, meta):
    margin_x = 15 * mm
    avail_w = page_w - 2 * margin_x
    top_y = page_h - 15 * mm

    # Bold Red Wall Flyer Header
    c.setFillColor(colors.HexColor("#b91c1c"))  # Bold Crimson Red
    c.rect(margin_x, top_y - 22 * mm, avail_w, 22 * mm, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(page_w / 2, top_y - 12 * mm, "MISSING PERSON")

    c.setFillColor(colors.HexColor("#fef08a"))  # Light Yellow
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(page_w / 2, top_y - 18 * mm, "PLEASE HELP US BRING THEM HOME   •   SHARE THIS NOTICE")

    # Large Center Real Photo (High Priority)
    photo_box_top = top_y - 25 * mm
    photo_h = 100 * mm
    photo_w = 90 * mm
    photo_x = (page_w - photo_w) / 2
    photo_y = photo_box_top - photo_h

    draw_fitted_image(c, primary_face.get('image_path', ''), photo_x, photo_y, photo_w, photo_h,
                      label="LAST KNOWN / RECENT PHOTOGRAPH", label_bg=colors.HexColor("#b91c1c"),
                      border_color=colors.HexColor("#b91c1c"), border_width=2.5)

    # Full Name Banner
    name_y = photo_y - 14 * mm
    name_str = (data.get('full_name') or 'NAME OF MISSING PERSON').upper()
    c.setFillColor(colors.HexColor("#b91c1c"))
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(page_w / 2, name_y, name_str)

    # Critical Missing Details Bar (Missing Since, Last Seen)
    crit_y = name_y - 18 * mm
    crit_h = 14 * mm
    c.setFillColor(colors.HexColor("#fef2f2"))  # Light pink-red tint
    c.setStrokeColor(colors.HexColor("#fca5a5"))
    c.roundRect(margin_x, crit_y, avail_w, crit_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#991b1b"))
    c.setFont("Helvetica-Bold", 8.5)
    missing_date = data.get('date_missing') or 'RECENT'
    loc = data.get('location') or 'LOCATION NOT SPECIFIED'
    c.drawString(margin_x + 5 * mm, crit_y + 4.5 * mm, f"MISSING SINCE: {missing_date}    |    LAST SEEN: {loc}")

    # Physical Attributes Grid
    attr_y = crit_y - 24 * mm
    attr_h = 20 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, attr_y, avail_w, attr_h, 4, fill=1, stroke=1)

    age_val = data.get('age', 'Unknown')
    curr_age = data.get('current_age')
    if curr_age:
        age_val = f"{age_val} (Now ~{curr_age})"

    attrs = [
        ("AGE", age_val),
        ("GENDER", data.get('gender', 'Unknown')),
        ("HEIGHT", data.get('height', 'Unknown')),
        ("BUILD", data.get('build', 'Unknown')),
        ("HAIR", data.get('hair', 'Unknown')),
        ("EYES", data.get('eyes', 'Unknown'))
    ]
    col_w = avail_w / 6.0
    for idx, (k, v) in enumerate(attrs):
        cx = margin_x + idx * col_w
        if idx > 0:
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.line(cx, attr_y + 2 * mm, cx, attr_y + attr_h - 2 * mm)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(cx + col_w / 2, attr_y + attr_h - 6 * mm, k)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(cx + col_w / 2, attr_y + 4.5 * mm, str(v)[:14])

    # Clothing & Distinguishing Marks
    extra_y = attr_y - 22 * mm
    extra_h = 18 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, extra_y, avail_w, extra_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#b91c1c"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(margin_x + 4 * mm, extra_y + extra_h - 5 * mm, "CLOTHING & IDENTIFYING MARKS:")

    cloth_info = f"Clothing: {data.get('clothing', 'Not specified')}."
    marks_info = data.get('identifying_marks')
    if marks_info:
        cloth_info += f"  Marks/Scars: {marks_info}."
    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica", 7.8)
    c.drawString(margin_x + 4 * mm, extra_y + extra_h - 11 * mm, cloth_info[:120])
    if len(cloth_info) > 120:
        c.drawString(margin_x + 4 * mm, extra_y + extra_h - 15.5 * mm, cloth_info[120:240])

    # Giant Emergency Contact Callout Box
    contact_h = 28 * mm
    contact_y = extra_y - contact_h - 5 * mm
    draw_contact_box(c, margin_x, contact_y, avail_w, contact_h,
                     phone=data.get('phone', ''),
                     alt_phone=data.get('alt_phone', ''),
                     contact_person=data.get('contact_person', ''),
                     case_no=data.get('case_number', ''),
                     agency=data.get('police_station', ''),
                     style="urgent")
    # NO generated disclaimer required for real photo missing person flyer!


# =========================================================================
# TEMPLATE 5: MISSING PERSON — REAL PHOTO + PROBABLE APPEARANCES
# Purpose: Long-term missing person flyer with aged / altered hypotheses.
# =========================================================================
def render_template_5(c, page_w, page_h, primary_face, addl_faces, data, meta):
    margin_x = 15 * mm
    avail_w = page_w - 2 * margin_x
    top_y = page_h - 15 * mm

    # Bold Header Banner
    c.setFillColor(colors.HexColor("#7f1d1d"))  # Dark Crimson
    c.rect(margin_x, top_y - 20 * mm, avail_w, 20 * mm, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_w / 2, top_y - 10.5 * mm, "MISSING PERSON — LONG-TERM SEARCH")

    c.setFillColor(colors.HexColor("#fef08a"))
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(page_w / 2, top_y - 16.5 * mm, "LAST KNOWN PHOTOGRAPH AND COMPUTER-GENERATED PROBABLE CURRENT APPEARANCES")

    # Visual Section: Last Known Real Photo (Left) + Probable Appearances Grid (Right)
    visual_top = top_y - 23 * mm
    visual_h = 88 * mm
    real_w = 75 * mm
    real_x = margin_x
    real_y = visual_top - visual_h

    # Left: Last Known Real Photo
    draw_fitted_image(c, primary_face.get('image_path', ''), real_x, real_y, real_w, visual_h,
                      label="LAST KNOWN REAL PHOTOGRAPH", label_bg=colors.HexColor("#7f1d1d"),
                      border_color=colors.HexColor("#7f1d1d"), border_width=2.5)

    # Right: Probable Appearances
    prob_x = real_x + real_w + 5 * mm
    prob_w = avail_w - real_w - 5 * mm

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(prob_x, visual_top - 4 * mm, "PROBABLE CURRENT APPEARANCES:")

    sub_faces = addl_faces[:4]
    n_sub = len(sub_faces)

    if n_sub > 0:
        sub_top = visual_top - 6 * mm
        sub_h = visual_h - 6 * mm
        rows = 2 if n_sub > 2 else 1
        cols = 2 if n_sub > 1 else 1
        cw = (prob_w - (cols - 1) * 3 * mm) / cols
        ch = (sub_h - (rows - 1) * 3 * mm) / rows

        for idx, sf in enumerate(sub_faces):
            r = idx // cols
            col = idx % cols
            cell_x = prob_x + col * (cw + 3 * mm)
            cell_y = sub_top - (r + 1) * ch - r * 3 * mm
            lbl = sf.get('label') or f"Probable Appearance {idx+1}"
            draw_fitted_image(c, sf.get('image_path', ''), cell_x, cell_y, cw, ch,
                              label="COMPUTER-GENERATED PROBABLE APPEARANCE", label_bg=colors.HexColor("#0f172a"),
                              border_color=colors.HexColor("#94a3b8"), border_width=1)
    else:
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.roundRect(prob_x, real_y, prob_w, visual_h - 6 * mm, 4, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(prob_x + prob_w / 2, real_y + visual_h / 2, "Select probable appearances to display.")

    # Name Banner
    name_y = real_y - 12 * mm
    name_str = (data.get('full_name') or 'NAME OF MISSING PERSON').upper()
    c.setFillColor(colors.HexColor("#7f1d1d"))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(page_w / 2, name_y, name_str)

    # Missing & Aging Timeline Bar
    time_y = name_y - 16 * mm
    time_h = 13 * mm
    c.setFillColor(colors.HexColor("#fef2f2"))
    c.setStrokeColor(colors.HexColor("#fca5a5"))
    c.roundRect(margin_x, time_y, avail_w, time_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#991b1b"))
    c.setFont("Helvetica-Bold", 8)
    age_str = f"AGE WHEN MISSING: {data.get('age', 'N/A')}"
    if data.get('current_age'):
        age_str += f"   |   ESTIMATED CURRENT AGE: ~{data.get('current_age')}"
    age_str += f"   |   MISSING SINCE: {data.get('date_missing', 'N/A')}"
    c.drawCentredString(page_w / 2, time_y + 4.5 * mm, age_str)

    # Attributes Table
    attr_y = time_y - 22 * mm
    attr_h = 18 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, attr_y, avail_w, attr_h, 4, fill=1, stroke=1)

    attrs = [
        ("GENDER", data.get('gender', 'Unknown')),
        ("HEIGHT", data.get('height', 'Unknown')),
        ("BUILD", data.get('build', 'Unknown')),
        ("HAIR", data.get('hair', 'Unknown')),
        ("EYES", data.get('eyes', 'Unknown')),
        ("MARKS", data.get('identifying_marks', 'None Recorded'))
    ]
    col_w = avail_w / 6.0
    for idx, (k, v) in enumerate(attrs):
        cx = margin_x + idx * col_w
        if idx > 0:
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.line(cx, attr_y + 2 * mm, cx, attr_y + attr_h - 2 * mm)
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica-Bold", 6.8)
        c.drawCentredString(cx + col_w / 2, attr_y + attr_h - 5.5 * mm, k)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8.2)
        c.drawCentredString(cx + col_w / 2, attr_y + 4 * mm, str(v)[:14])

    # Narrative Notes
    narr_y = attr_y - 20 * mm
    narr_h = 16 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(margin_x, narr_y, avail_w, narr_h, 4, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#7f1d1d"))
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(margin_x + 4 * mm, narr_y + narr_h - 4.5 * mm, "CIRCUMSTANCES & INVESTIGATIVE DETAILS:")
    info_text = data.get('description') or data.get('notes') or f"Last seen location: {data.get('location', 'Unknown')}."
    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica", 7.5)
    c.drawString(margin_x + 4 * mm, narr_y + narr_h - 10 * mm, info_text[:120])
    if len(info_text) > 120:
        c.drawString(margin_x + 4 * mm, narr_y + narr_h - 14.5 * mm, info_text[120:240])

    # Urgent Contact Callout Box
    contact_h = 26 * mm
    contact_y = narr_y - contact_h - 4 * mm
    draw_contact_box(c, margin_x, contact_y, avail_w, contact_h,
                     phone=data.get('phone', ''),
                     alt_phone=data.get('alt_phone', ''),
                     contact_person=data.get('contact_person', ''),
                     case_no=data.get('case_number', ''),
                     agency=data.get('police_station', ''),
                     style="urgent")

    # Mandatory Probable Appearance Disclaimer
    draw_disclaimer_footer(c, page_w, 7 * mm, avail_w, text_type="probable")


# =========================================================================
# MAIN ENTRY POINT: generate_poster_pdf
# =========================================================================
def generate_poster_pdf(primary_face, addl_faces=None, template_id=1, person_data=None, metadata=None, orientation='portrait', output_dir='static/generated_posters'):
    """
    Renders an A4 PDF wall poster for the selected person and images.
    Supports all 5 redesigned public display templates.
    """
    start_time = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    if addl_faces is None:
        addl_faces = []
    if person_data is None:
        person_data = {}
    if metadata is None:
        metadata = {}

    meta = {
        'title': metadata.get('title') or person_data.get('custom_title') or 'PERSON INFORMATION',
        'ref_no': metadata.get('ref_no') or person_data.get('case_number') or f"TE-{time.strftime('%Y')}-001",
        'date': metadata.get('date') or person_data.get('date_missing') or time.strftime('%d/%m/%Y'),
        'notes': metadata.get('notes') or person_data.get('description') or ''
    }

    # True A4 Page Size
    if orientation == 'landscape':
        page_size = landscape(A4)
        page_w, page_h = page_size
    else:
        page_size = A4
        page_w, page_h = page_size

    uid = uuid.uuid4().hex[:10]
    pdf_filename = f"poster_{uid}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    c = canvas.Canvas(pdf_path, pagesize=page_size, pageCompression=0)
    c.setTitle(meta['title'])
    c.setAuthor("ThirdEye Forensic Poster Engine")
    c.setSubject("Public Display Wall Poster")

    if template_id == 1:
        render_template_1(c, page_w, page_h, primary_face, addl_faces, person_data, meta)
    elif template_id == 2:
        render_template_2(c, page_w, page_h, primary_face, addl_faces, person_data, meta)
    elif template_id == 3:
        render_template_3(c, page_w, page_h, primary_face, addl_faces, person_data, meta)
    elif template_id == 4:
        render_template_4(c, page_w, page_h, primary_face, addl_faces, person_data, meta)
    elif template_id == 5:
        render_template_5(c, page_w, page_h, primary_face, addl_faces, person_data, meta)
    else:
        render_template_1(c, page_w, page_h, primary_face, addl_faces, person_data, meta)

    c.showPage()
    c.save()

    elapsed = round(time.perf_counter() - start_time, 3)
    file_size_kb = round(os.path.getsize(pdf_path) / 1024.0, 1)

    return {
        "success": True,
        "pdf_filename": pdf_filename,
        "pdf_path": pdf_path,
        "template_id": template_id,
        "orientation": orientation,
        "processing_time": elapsed,
        "file_size_kb": file_size_kb,
        "dimensions_mm": [297, 210] if orientation == 'landscape' else [210, 297],
        "primary_image": primary_face.get('image_path'),
        "additional_count": len(addl_faces)
    }
