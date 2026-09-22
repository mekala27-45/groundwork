"""Authors the Meridian Coaching demo workspace's four source documents
from structured content defined in this file, not typed once into a PDF
editor and forgotten: every fact any eval question in evalset/questions.yaml
depends on traces back to a paragraph here.

Meridian Coaching is fictional, invented for this project, stated as such
in README.md. Every figure (prices, session counts, response windows) is
made up but internally consistent across all four documents, exactly the
way a real small business's own materials would be. No testimonials,
reviews, or client quotes appear anywhere, per the day 5 prompt's hard
constraint against fabricated social proof.

Run with: uv run python scripts/build_demo_documents.py
"""

from __future__ import annotations

from pathlib import Path

from pdf_builder import DocBuilder

OUT_DIR = Path(__file__).resolve().parent.parent / "evalset" / "meridian"


def build_methodology() -> None:
    b = DocBuilder()
    b.h1("The Meridian Method")
    b.paragraph(
        "Meridian Coaching works from a single framework across every engagement, called the "
        "Meridian Method. It was built on the observation that most coaching relationships "
        "stall not because the client lacks motivation, but because nobody separated what the "
        "client actually wants from what they believe they are supposed to want. The Meridian "
        "Method starts there, before any goal setting happens."
    )
    b.h2("Four Pillars")
    b.paragraph(
        "Clarity comes first: naming the outcome the client actually wants, in their own words, "
        "separated from inherited expectations from a parent, a former manager, or a prior "
        "version of themselves. A coach cannot help a client reach a goal that was never really "
        "theirs."
    )
    b.paragraph(
        "Capacity is an honest accounting of the time, energy, and money the client actually has "
        "available, not the amount they wish they had. Plans built against wished-for capacity "
        "are the single most common reason coaching engagements fail to produce change."
    )
    b.paragraph(
        "Constraints means naming the real obstacle, which is usually not the one the client "
        "leads with. A client who says they lack discipline is often, on closer examination, "
        "dealing with an unclear priority, a schedule conflict, or a fear they have not stated "
        "out loud yet."
    )
    b.paragraph(
        "Cadence is the sustainable pace of action and accountability the client can actually "
        "keep up, week over week, without burning out by the six week mark. Meridian coaches are "
        "trained to slow a client down who is setting a cadence they cannot sustain, since a "
        "missed pace does more damage to momentum than a slower one ever would."
    )
    b.h2("The Baseline Assessment")
    b.paragraph(
        "Every client completes the Meridian Baseline Assessment before their first full "
        "session. It scores five domains on a scale of one to ten: career, relationships, "
        "health, finances, and sense of purpose. The Baseline Assessment is re-administered at "
        "the midpoint of an engagement and again at the end, so progress is measured against a "
        "client's own starting point rather than compared to anyone else's."
    )
    b.h2("How Sessions Work")
    b.paragraph(
        "Sessions run fifty minutes and take place over video by default, with phone available "
        "on request. Meridian does not offer in-person sessions at any location; the practice is "
        "fully remote. With a client's consent, a session recording is kept available to that "
        "client for thirty days afterward, so they can revisit anything discussed."
    )
    b.h2("Who Delivers the Coaching")
    b.paragraph(
        "Meridian Coaching is a small practice of three coaches, each credentialed through the "
        "International Coaching Federation and trained directly in the Meridian Method before "
        "taking on clients. A new client is matched with one primary coach based on their "
        "Baseline Assessment results and the goals discussed on the discovery call, and works "
        "with that same coach for the length of the engagement."
    )
    b.save(OUT_DIR / "methodology.pdf")


def build_packages() -> None:
    b = DocBuilder()
    b.h1("Coaching Packages")
    b.paragraph(
        "Meridian offers three ongoing packages and a single session option with no commitment. "
        "Every package includes the Meridian Baseline Assessment at intake and is delivered by "
        "the same primary coach for the length of the engagement."
    )
    b.h2("Foundations")
    b.paragraph(
        "Foundations costs four hundred fifty dollars per month and includes two fifty-minute "
        "sessions per month, the Baseline Assessment at intake, and email support between "
        "sessions with a forty-eight hour response window. Foundations is billed month to month "
        "with no minimum commitment, and is the right starting point for a client who wants "
        "structure without a heavy weekly cadence."
    )
    b.h2("Momentum")
    b.paragraph(
        "Momentum costs eight hundred fifty dollars per month and includes four fifty-minute "
        "sessions per month, delivered weekly. It includes the Baseline Assessment at intake and "
        "a midpoint reassessment, priority email support with a twenty-four hour response "
        "window, and one fifteen-minute check-in call included each month between full sessions."
    )
    b.h2("Executive")
    b.paragraph(
        "Executive costs one thousand six hundred fifty dollars per month and includes four "
        "fifty-minute sessions per month plus unlimited asynchronous messaging with the assigned "
        "coach. It adds a quarterly ninety-minute deep-dive session, a personalized "
        "action-tracking dashboard, and a dedicated senior coach rather than assignment from the "
        "general roster."
    )
    b.h2("Add-Ons and Single Sessions")
    b.paragraph(
        "A single session with no ongoing package costs two hundred seventy-five dollars. "
        "Couples coaching can be added to any package for an additional one hundred fifty "
        "dollars per month, covering joint sessions in addition to the individual sessions "
        "already included."
    )
    b.h2("Policies That Apply to Every Package")
    b.paragraph(
        "New clients on any package are covered by a fourteen day money-back guarantee: if a "
        "client is not satisfied after their first session, the first month is refunded in full "
        "with no reason required. Outside that first-month window, packages are non-refundable "
        "but can be cancelled at any time with thirty days of notice."
    )
    b.paragraph(
        "A package can be paused for up to six weeks within any twelve-month engagement, "
        "provided the pause is requested at least five business days in advance. Momentum and "
        "Executive both carry a three-month minimum commitment from the start date; Foundations "
        "has no minimum and can be cancelled following the standard thirty-day notice policy at "
        "any time."
    )
    b.save(OUT_DIR / "packages.pdf")


def build_faq() -> None:
    b = DocBuilder()
    b.h1("Frequently Asked Questions")
    b.h2("Getting Started")
    b.paragraph(
        "Every new relationship with Meridian begins with a free twenty-minute discovery call. "
        "There is no obligation attached to the call: it exists so a prospective client and a "
        "coach can confirm the fit is right before any payment changes hands."
    )
    b.h2("Format and Scheduling")
    b.paragraph(
        "Meridian is a fully remote practice with no physical office. Sessions happen over video "
        "by default, with phone available on request. Coaches are available for scheduling "
        "Monday through Friday from eight in the morning to seven in the evening Eastern time, "
        "and Saturday from nine in the morning to one in the afternoon Eastern time. Meridian "
        "does not offer Sunday availability."
    )
    b.paragraph(
        "Individual sessions can be rescheduled at no charge with at least twenty-four hours of "
        "notice. A session cancelled with less than twenty-four hours of notice is forfeited."
    )
    b.h2("Who Coaching Is For")
    b.paragraph(
        "Meridian works primarily with working professionals navigating a career transition, "
        "recovering from burnout, or facing a major life decision. Coaching is not a substitute "
        "for therapy or mental health treatment, and the intake process, including the Baseline "
        "Assessment, is used to screen for concerns that call for a referral to a licensed "
        "mental health professional instead."
    )
    b.h2("Confidentiality")
    b.paragraph(
        "Session notes are stored encrypted and are never shared with a client's employer, even "
        "when that employer is the one paying for a corporate-sponsored package, except with the "
        "client's written consent or where disclosure is required by law."
    )
    b.save(OUT_DIR / "faq.pdf")


def build_onboarding() -> None:
    b = DocBuilder()
    b.h1("Getting Started: The Onboarding Process")
    b.paragraph(
        "Meridian's onboarding runs in five stages, from a prospective client's first call "
        "through their first structured progress review. Most clients move from the discovery "
        "call to their kickoff session in seven to ten days."
    )
    b.h2("Stage One: Discovery Call")
    b.paragraph(
        "A free twenty-minute call with no obligation. The coach and the prospective client "
        "discuss what the client wants to work on, the coach explains the Meridian Method, and "
        "both sides confirm the fit is right before anything is booked."
    )
    b.h2("Stage Two: Baseline Assessment")
    b.paragraph(
        "Within three days of the discovery call, the client completes the Meridian Baseline "
        "Assessment, scoring career, relationships, health, finances, and sense of purpose on a "
        "scale of one to ten. The results shape which coach the client is matched with next."
    )
    b.h2("Stage Three: Coach Matching and Package Selection")
    b.paragraph(
        "Using the Baseline Assessment results and the goals discussed on the discovery call, "
        "the client is matched with one of Meridian's three coaches and selects a package: "
        "Foundations, Momentum, or Executive."
    )
    b.h2("Stage Four: Kickoff Session")
    b.paragraph(
        "The client's first full fifty-minute session. The coach and client set specific, "
        "named outcomes for the engagement and agree on a session cadence the client can "
        "actually sustain, in line with the Meridian Method's fourth pillar."
    )
    b.h2("Stage Five: First 30-Day Check-In")
    b.paragraph(
        "A structured review, thirty days after the kickoff session, comparing progress against "
        "the outcomes set at kickoff. The coach and client adjust the plan together if the "
        "original pace or goals no longer fit what the client has learned about themselves in "
        "the first month."
    )
    b.save(OUT_DIR / "onboarding.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_methodology()
    build_packages()
    build_faq()
    build_onboarding()
    for pdf in sorted(OUT_DIR.glob("*.pdf")):
        print(pdf.relative_to(OUT_DIR.parent.parent))


if __name__ == "__main__":
    main()
