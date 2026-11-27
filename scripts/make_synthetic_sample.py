"""Builds a SYNTHETIC multi-page judgment PDF for smoke tests/demo. It is fiction, not a real judgment."""
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

PAGES = [
 ["IN THE SUPREME COURT OF INDIA", "CIVIL APPELLATE JURISDICTION", "CIVIL APPEAL NO. 4521 OF 2019",
  "Ramesh Kumar Sharma v. State of Madhya Pradesh", "(SYNTHETIC SAMPLE - NOT A REAL JUDGMENT)",
  "The appellant, Ramesh Kumar Sharma, was appointed as a clerk in the Revenue Department on 12 March 2004. "
  "On 5 June 2012, the department issued a show-cause notice alleging misappropriation of Rs. 45,000. "
  "The appellant filed a writ petition before the High Court challenging his dismissal from service. "
  "The High Court dismissed the writ petition on 18 January 2018. Aggrieved, the appellant has preferred this appeal.",
  "The brief facts are that the inquiry officer submitted his report on 9 August 2013 without examining two material witnesses. "
  "The disciplinary authority nonetheless accepted the report and passed an order of dismissal on 3 October 2013."],
 ["Leave granted. Heard the learned counsel for the parties.",
  "Whether the dismissal order is vitiated for non-compliance with the principles of natural justice? "
  "Whether the High Court was justified in declining interference under Article 226 of the Constitution?",
  "Learned counsel for the appellant submitted that the inquiry was conducted in violation of the rules. "
  "It was further contended that the appellant was denied an opportunity to cross-examine the witnesses. "
  "The appellant also urged that the punishment of dismissal is disproportionate to the alleged misconduct."],
 ["Learned counsel for the respondent State submitted that the inquiry was fair and the appellant participated at every stage. "
  "It was urged that the High Court rightly declined to interfere with findings of fact. "
  "The State further contended that the misappropriation of public money warrants dismissal.",
  "We have considered the submissions of the parties. It is well settled that an inquiry must afford a reasonable opportunity "
  "to the delinquent employee. In our view, the failure to examine material witnesses is not a mere technical lapse. "
  "We find that the report of the inquiry officer is not based on any evidence on the record. "
  "This Court has held in several decisions that a dismissal founded on a defective inquiry cannot be sustained."],
 ["Therefore, the order of dismissal dated 3 October 2013 cannot be sustained. "
  "In the result, the appeal is allowed. The judgment of the High Court dated 18 January 2018 is set aside. "
  "We direct the respondent to reinstate the appellant within four weeks. There shall be no order as to costs."],
]


def build(path):
    ss = getSampleStyleSheet()
    def header():
        return [Paragraph("IN THE SUPREME COURT OF INDIA - SYNTHETIC SAMPLE", ss["Normal"]), Spacer(1, 6)]
    story = []
    for i, page in enumerate(PAGES, 1):
        story += header()
        for para in page:
            story += [Paragraph(para, ss["Normal"]), Spacer(1, 8)]
        story += [Spacer(1, 6), Paragraph(f"Page {i}", ss["Normal"])]
        if i < len(PAGES):
            story.append(PageBreak())
    SimpleDocTemplate(path, pagesize=A4).build(story)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_sample.pdf")
