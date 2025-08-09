
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
from io import BytesIO

logger = logging.getLogger(__name__)


# --- FONCTION 1 : GÉNÉRATION DU RÉSUMÉ DE L'EMPLOI DU TEMPS (INCHANGÉE) ---
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
                hotel_name = city_plan_item_schedule["hotel"][0].get('nom', "Hôtel recommandé")

            schedule_markdown_list.append(f"- 🏨 Séjour à : {hotel_name}\n")
            schedule_story_pdf.append(Paragraph(f"• Séjour à : {hotel_name}", schedule_normal_style))

            activities_to_list_for_day = [act for act in ordered_points_this_day if act.get('type') != 'hotel' and 'Gastronomique' not in act.get('type','')]

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


# --- FONCTION 2 : GÉNÉRATION DU PDF COMPLET (NOUVELLE VERSION INTELLIGENTE) ---
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
    italic_style = ParagraphStyle('ItalicUTF8', parent=styles['Italic'], fontName=f'{default_font}-Oblique' if default_font=='Helvetica' else 'Times-Italic', fontSize=9)
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

    grand_total_hotel_cost = 0
    grand_total_activity_cost = 0

    current_day = 1
    for i, city_data in enumerate(trip_plan_data):
        if not city_data.get("jours_alloues"): continue
        story.append(Paragraph(f"Plan pour {city_data['ville']} ({city_data['jours_alloues']} jour(s))", h2_style))
        
        total_hotel_cost_for_city = 0
        suggested_hotels = city_data.get("hotel", [])
        if suggested_hotels:
            recommended_hotel = suggested_hotels[0]
            story.append(Paragraph("Hébergement Recommandé :", h3_style))
            
            # Logique robuste pour trouver le prix, peu importe la clé ('budget_estime' ou 'price_per_night')
            price = recommended_hotel.get('budget_estime') or recommended_hotel.get('price_per_night', 0)
            if not isinstance(price, (int, float)): price = 0

            name = recommended_hotel.get('nom', 'N/A')
            rating = recommended_hotel.get('rating', 'N/A')
            story.append(Paragraph(f"• <b>{name}</b> (Note: {rating}/10)", normal_style))
            story.append(Paragraph(f"  Prix estimé par nuit : <b>{price:.2f} MAD</b>", normal_style))
            
            if recommended_hotel.get('booking_link') not in [None, 'N/A']:
                story.append(Paragraph(f"  <a href='{recommended_hotel['booking_link']}'>Lien de réservation</a>", link_style))
            story.append(Paragraph("  <i>D'autres options sont disponibles sur le site web.</i>", italic_style))
            
            num_days_in_city = city_data.get('jours_alloues', 1)
            total_hotel_cost_for_city = price * num_days_in_city
            grand_total_hotel_cost += total_hotel_cost_for_city
            story.append(Spacer(1, 0.1*inch))
        
        for day_plan in city_data.get("activites_par_jour_optimisees", []):
            story.append(Paragraph(f"<b>Activités - Jour {current_day}:</b>", bold_style))
            activities = [p for p in day_plan if p.get('type') != 'hotel' and 'Gastronomique' not in p.get('type', '')]
            
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
                        Paragraph(str(k+1), normal_style), Paragraph(point.get('nom', 'N/A'), normal_style),
                        Paragraph(budget_str, normal_style), Paragraph(total_cost_str, normal_style),
                        Paragraph(str(point.get('duree_estimee', 'N/A')), normal_style)
                    ])
                act_table = Table(table_data, colWidths=[0.3*inch, 3*inch, 1*inch, 1*inch, 1.2*inch])
                act_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightcyan), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
                story.append(act_table)
            
            story.append(Spacer(1, 0.1*inch)); current_day +=1
        
        city_activity_budget = city_data.get('budget_activites_depense', 0)
        grand_total_activity_cost += city_activity_budget
        total_city_budget = total_hotel_cost_for_city + city_activity_budget

        story.append(Paragraph("<b>Résumé budgétaire pour " + city_data['ville'] + "</b>", h3_style))
        budget_summary_data = [
            [Paragraph("Coût estimé des hôtels:", normal_style), Paragraph(f"{total_hotel_cost_for_city:.2f} MAD", normal_style)],
            [Paragraph("Budget estimé (Activités & Repas):", normal_style), Paragraph(f"{city_activity_budget:.2f} MAD", normal_style)],
            [Paragraph("<b>Total estimé pour la ville:</b>", bold_style), Paragraph(f"<b>{total_city_budget:.2f} MAD</b>", bold_style)],
        ]
        budget_summary_table = Table(budget_summary_data, colWidths=[2.5*inch, 2*inch])
        budget_summary_table.setStyle(TableStyle([('ALIGN', (1,0), (1,-1), 'RIGHT'), ('GRID', (0,0), (-1,-1), 0.25, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(budget_summary_table)
        story.append(Spacer(1, 0.2*inch))

        if i < len(trip_plan_data) - 1: story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Résumé Budgétaire Global du Voyage", h2_style))
    story.append(Spacer(1, 0.2*inch))

    grand_total_budget = grand_total_hotel_cost + grand_total_activity_cost
    budget_per_person = grand_total_budget / num_persons if num_persons > 0 else 0

    story.append(Paragraph(f"Ce résumé est une estimation pour <b>{num_persons} personne(s)</b>.", normal_style))
    story.append(Spacer(1, 0.2*inch))

    final_budget_data = [
        [Paragraph('<b>Poste de dépense</b>', bold_style), Paragraph('<b>Coût total estimé</b>', bold_style)],
        [Paragraph('Coût total Hôtels', normal_style), Paragraph(f'{grand_total_hotel_cost:.2f} MAD', normal_style)],
        [Paragraph('Coût total Activités & Repas', normal_style), Paragraph(f'{grand_total_activity_cost:.2f} MAD', normal_style)],
        [Paragraph('<b>BUDGET TOTAL ESTIMÉ</b>', bold_style), Paragraph(f'<b>{grand_total_budget:.2f} MAD</b>', bold_style)],
        [Paragraph('<b>Budget estimé par personne</b>', bold_style), Paragraph(f'<b>{budget_per_person:.2f} MAD</b>', bold_style)],
    ]

    final_budget_table = Table(final_budget_data, colWidths=[3.5*inch, 3*inch], rowHeights=0.4*inch)
    final_budget_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C5D99")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), f'{default_font}-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (1,2), colors.HexColor("#EAF1FA")),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,3), (-1,4), colors.HexColor("#C3D69B")),
        ('FONTSIZE', (0,3), (-1,4), 11),
    ]))
    story.append(final_budget_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<i>*Note: Ces montants sont des estimations basées sur les données disponibles et vos préférences. Les coûts réels peuvent varier. Le budget ne comprend pas les transports inter-villes, les repas non listés et les dépenses personnelles.</i>", italic_style))

    if schedule_pdf_content_objects:
        story.append(PageBreak()); story.extend(schedule_pdf_content_objects)
        
    try:
        doc.build(story)
        logger.info("Le fichier PDF a été généré avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la construction du PDF : {e}", exc_info=True); raise

# --- FONCTION DE GÉNÉRATION DU PDF VOYAGE (INCHANGÉE) ---
def generate_voyage_pdf_django(output_buffer, voyage_obj):
    doc = SimpleDocTemplate(output_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['h1'], fontSize=24, spaceAfter=20, alignment=1)
    author_style = ParagraphStyle('Author', parent=styles['h3'], fontSize=12, alignment=1, spaceAfter=20, fontName='Helvetica-Oblique')
    day_title_style = ParagraphStyle('DayTitle', parent=styles['h2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#222255"))
    story_style = ParagraphStyle('Story', parent=styles['Normal'], fontSize=11, leading=14, spaceAfter=12)

    story = []
    
    story.append(Paragraph(voyage_obj.title, title_style))
    story.append(Paragraph(f"par {voyage_obj.author.username}", author_style))
    
    journal_entries = voyage_obj.journal_entries.all().order_by('day_number')
    for entry in journal_entries:
        story.append(Paragraph(f"Jour {entry.day_number}: {entry.title}", day_title_style))
        story_text = entry.story.replace('\n', '<br/>')
        story.append(Paragraph(story_text, story_style))
        story.append(Spacer(1, 0.2 * inch))

    try:
        doc.build(story)
        logger.info(f"PDF pour le voyage '{voyage_obj.title}' généré avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la construction du PDF : {e}", exc_info=True); raise