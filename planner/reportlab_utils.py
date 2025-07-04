# planner/reportlab_utils.py

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


# --- FONCTION 1 : GÉNÉRATION DU RÉSUMÉ DE L'EMPLOI DU TEMPS ---
def generate_schedule_content_objects_django(trip_plan_data, num_total_days):
    schedule_story_pdf = []
    schedule_markdown_list = ["### 📅 Emploi du Temps Suggéré\n"]

    styles = getSampleStyleSheet()
    default_font_schedule = 'Helvetica'
    try:
        Paragraph("test", ParagraphStyle('test_bold', fontName='Helvetica-Bold'))
    except Exception:
        default_font_schedule = 'Times-Roman'
        logger.warning("Police Helvetica non trouvée pour l'emploi du temps PDF, utilisation de Times-Roman.")

    schedule_normal_style = ParagraphStyle('ScheduleNormal', parent=styles['Normal'], fontName=default_font_schedule, fontSize=9, leading=11)
    schedule_bold_style = ParagraphStyle('ScheduleBold', parent=styles['Normal'], fontName=f'{default_font_schedule}-Bold', fontSize=10, leading=12, spaceBefore=6)
    schedule_h3_style = ParagraphStyle('ScheduleH3', parent=styles['h3'], fontName=f'{default_font_schedule}-Bold', fontSize=11, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#222255"))

    schedule_story_pdf.append(Paragraph("<b>Emploi du Temps Suggéré</b>", schedule_h3_style))
    schedule_story_pdf.append(Spacer(1, 0.1*inch))

    current_day_counter = 1
    for city_plan_item_schedule in trip_plan_data:
        city_name_schedule = city_plan_item_schedule['ville']
        daily_activity_lists_for_schedule = city_plan_item_schedule.get('activites_par_jour_optimisees', [])

        for day_idx, ordered_points_this_day in enumerate(daily_activity_lists_for_schedule):
            if current_day_counter > num_total_days: break

            schedule_markdown_list.append(f"**Jour {current_day_counter} : {city_name_schedule}**\n")
            schedule_story_pdf.append(Paragraph(f"<b>Jour {current_day_counter} : {city_name_schedule}</b>", schedule_bold_style))
            
            hotel_name = "Hôtel non spécifié"
            if city_plan_item_schedule.get("hotel") and city_plan_item_schedule["hotel"]:
                hotel_name = city_plan_item_schedule["hotel"][0].get('name', "Hôtel recommandé")

            schedule_markdown_list.append(f"- 🏨 Séjour à : {hotel_name}\n")
            schedule_story_pdf.append(Paragraph(f"• Séjour à : {hotel_name}", schedule_normal_style))

            activities_to_list_for_day = [act for act in ordered_points_this_day if act.get('type') != 'hotel']

            if activities_to_list_for_day:
                for act_schedule in activities_to_list_for_day:
                    activity_name_schedule = act_schedule.get('nom', 'Activité')
                    activity_type_schedule = act_schedule.get('type', 'N/A')
                    activity_duration_schedule = act_schedule.get('duree_estimee', 'N/A')
                    schedule_markdown_list.append(f"  - 🎯 {activity_name_schedule} ({activity_type_schedule}, Durée: {activity_duration_schedule})\n")
                    schedule_story_pdf.append(Paragraph(f"  • {activity_name_schedule} ({activity_type_schedule}, Durée: {activity_duration_schedule})", schedule_normal_style))
            else:
                schedule_markdown_list.append("  - Exploration libre / Détente / Repos à l'hôtel\n")
                schedule_story_pdf.append(Paragraph("  • Exploration libre / Détente / Repos à l'hôtel", schedule_normal_style))

            schedule_markdown_list.append("\n")
            schedule_story_pdf.append(Spacer(1, 0.05*inch))
            current_day_counter += 1

    return "".join(schedule_markdown_list), schedule_story_pdf


# --- FONCTION 2 : GÉNÉRATION DU PDF COMPLET (CORRIGÉE) ---
def generate_trip_pdf_django(output_buffer, trip_plan_data, trip_params, schedule_pdf_content_objects):
    doc = SimpleDocTemplate(output_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    story = []
    default_font = 'Helvetica'
    try:
        Paragraph("test", ParagraphStyle('test_bold', fontName='Helvetica-Bold'))
    except Exception:
        default_font = 'Times-Roman'
    
    normal_style = ParagraphStyle('NormalUTF8', parent=styles['Normal'], fontName=default_font, leading=14, fontSize=9)
    bold_style = ParagraphStyle('BoldNormalUTF8', parent=normal_style, fontName=f'{default_font}-Bold')
    title_style = ParagraphStyle('TitleUTF8', parent=styles['h1'], fontName=f'{default_font}-Bold', fontSize=18, spaceAfter=12, alignment=1)
    h2_style = ParagraphStyle('H2UTF8', parent=styles['h2'], fontName=f'{default_font}-Bold', fontSize=14, spaceBefore=10, spaceAfter=6)
    h3_style = ParagraphStyle('H3UTF8', parent=styles['h3'], fontName=f'{default_font}-Bold', fontSize=11, spaceBefore=8, spaceAfter=4)
    italic_style = ParagraphStyle('ItalicUTF8', parent=styles['Italic'], fontName=f'{default_font}-Oblique' if default_font=='Helvetica' else 'Times-Italic', fontSize=9, alignment=2)
    link_style = ParagraphStyle('LinkStyle', parent=styles['Normal'], textColor=colors.blue, fontName=default_font)

    story.append(Paragraph("Plan de Voyage Recommandé au Maroc", title_style))
    story.append(Paragraph(f"<i>Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>", italic_style))
    story.append(Spacer(1, 0.2*inch))
    
    num_persons_str = str(trip_params.get("Nombre de personnes", "1"))
    try:
        match = re.search(r'^\d+', num_persons_str)
        num_persons = int(match.group(0)) if match else 1
    except (ValueError, AttributeError):
        num_persons = 1

    story.append(Paragraph("<b>Paramètres du Voyage</b>", h2_style))
    param_data = [[Paragraph(f"<b>{key}</b>", bold_style), Paragraph(str(value), normal_style)] for key, value in trip_params.items()]
    param_table = Table(param_data, colWidths=[2.5*inch, 4*inch])
    param_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)]))
    story.append(param_table)
    story.append(Spacer(1, 0.3*inch))

    current_day = 1
    for i, city_data in enumerate(trip_plan_data):
        if not city_data.get("jours_alloues"): continue
        story.append(Paragraph(f"Plan pour {city_data['ville']} ({city_data['jours_alloues']} jour(s))", h2_style))
        
        hotel_info = city_data.get("hotel")[0] if city_data.get("hotel") else None
        if hotel_info:
            story.append(Paragraph("Hôtel Suggéré :", h3_style))
            name = hotel_info.get('name') or hotel_info.get('nom', 'N/A')
            price = hotel_info.get('price_per_night', 0)
            story.append(Paragraph(f"• {name} (Rating: {hotel_info.get('rating', 'N/A')}, Prix/nuit: {price:.2f} MAD)", normal_style))
            if hotel_info.get('booking_link') not in [None, 'N/A']:
                story.append(Paragraph(f"<a href='{hotel_info['booking_link']}'>Lien de réservation</a>", link_style))
            story.append(Spacer(1, 0.1*inch))

        for day_plan in city_data.get("activites_par_jour_optimisees", []):
            story.append(Paragraph(f"<b>Jour {current_day}:</b>", bold_style))
            activities = [p for p in day_plan if p.get('type') != 'hotel']
            
            if not activities:
                story.append(Paragraph("  • Repos / Exploration libre", normal_style))
            else:
                header = [Paragraph(f"<b>{h}</b>", bold_style) for h in ["#", "Activité", "Budget/pers.", "Coût Total", "Durée"]]
                table_data = [header]
                
                for k, point in enumerate(activities):
                    budget = point.get('budget_estime', 0)
                    if isinstance(budget, (int, float)) and pd.notna(budget):
                        total_cost = budget * num_persons
                        budget_str = f"{budget:.0f}"
                        total_cost_str = f"{total_cost:.0f}"
                    else:
                        budget_str, total_cost_str = "N/A", "N/A"

                    table_data.append([
                        Paragraph(str(k+1), normal_style),
                        Paragraph(point.get('nom', 'N/A'), normal_style),
                        Paragraph(budget_str, normal_style),
                        Paragraph(total_cost_str, normal_style),
                        Paragraph(str(point.get('duree_estimee', 'N/A')), normal_style)
                    ])
                
                act_table = Table(table_data, colWidths=[0.3*inch, 3*inch, 1*inch, 1*inch, 1.2*inch])
                act_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightcyan), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
                story.append(act_table)
            
            story.append(Spacer(1, 0.1*inch)); current_day +=1
        
        story.append(Paragraph(f"<i>Budget total activités estimé pour {city_data['ville']} ({num_persons} pers.): {city_data.get('budget_activites_depense', 0):.0f} MAD</i>", styles['Italic']))
        story.append(Spacer(1, 0.2*inch))
        if i < len(trip_plan_data) - 1: story.append(PageBreak())

    if schedule_pdf_content_objects:
        story.append(PageBreak()); story.extend(schedule_pdf_content_objects)
        
    try:
        doc.build(story)
        logger.info("Le fichier PDF a été généré avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la construction du PDF : {e}", exc_info=True); raise

# À la fin de planner/reportlab_utils.py

from io import BytesIO
from reportlab.lib.styles import ParagraphStyle

def generate_voyage_pdf_django(output_buffer, voyage_obj):
    """Génère un PDF simple pour un objet Voyage."""
    doc = SimpleDocTemplate(output_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Définir des styles personnalisés
    title_style = ParagraphStyle('Title', parent=styles['h1'], fontSize=24, spaceAfter=20, alignment=1)
    author_style = ParagraphStyle('Author', parent=styles['h3'], fontSize=12, alignment=1, spaceAfter=20, fontName='Helvetica-Oblique')
    day_title_style = ParagraphStyle('DayTitle', parent=styles['h2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#222255"))
    story_style = ParagraphStyle('Story', parent=styles['Normal'], fontSize=11, leading=14, spaceAfter=12)

    story = []
    
    # Titre et auteur
    story.append(Paragraph(voyage_obj.title, title_style))
    story.append(Paragraph(f"par {voyage_obj.author.username}", author_style))
    
    # Contenu par jour
    journal_entries = voyage_obj.journal_entries.all().order_by('day_number')
    for entry in journal_entries:
        story.append(Paragraph(f"Jour {entry.day_number}: {entry.title}", day_title_style))
        # Remplacer les sauts de ligne par des balises <br/> pour ReportLab
        story_text = entry.story.replace('\n', '<br/>')
        story.append(Paragraph(story_text, story_style))
        story.append(Spacer(1, 0.2 * inch))

    try:
        doc.build(story)
        logger.info(f"PDF pour le voyage '{voyage_obj.title}' généré avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la construction du PDF : {e}", exc_info=True); raise