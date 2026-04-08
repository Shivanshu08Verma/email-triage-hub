"""
Email dataset with ground-truth labels for the Email Triage Hub environment.
15 emails covering urgent, normal, low priority, and spam categories.
"""
from typing import List, Dict, Any

EMAILS: List[Dict[str, Any]] = [
    # URGENT
    {
        "email_id": "E001",
        "sender": "devops@techcorp.com",
        "subject": "URGENT: Production Database Down – All Services Offline",
        "body": (
            "Hi Team,\n\n"
            "Our production database cluster went down completely at 09:00 AM. "
            "All customer-facing services are offline.\n\n"
            "Error logs show: 'Connection refused – max connections exceeded'.\n\n"
            "Impact:\n"
            "  - ~50,000 active users cannot access the platform\n"
            "  - Estimated revenue loss: ~$10,000 / minute\n"
            "  - Outage duration so far: 12 minutes\n\n"
            "I have already tried restarting the connection pool but it did not help. "
            "Immediate action from a senior DBA is required.\n\n"
            "-- DevOps On-Call"
        ),
        "timestamp": "2024-01-15T09:12:00Z",
        "has_attachment": True,
        "true_priority": "urgent",
        "true_department": "IT",
        "is_spam": False,
        "needs_response": True,
    },
    {
        "email_id": "E002",
        "sender": "sarah.johnson@bigclient.com",
        "subject": "Contract Renewal – Need Signature TODAY or Deal Falls Through",
        "body": (
            "Dear Account Manager,\n\n"
            "I am reaching out urgently regarding our enterprise contract renewal. "
            "Our board approved the $2.4 M renewal yesterday, but your countersignature "
            "is required TODAY before 5 PM EST. If we do not receive the signed contract "
            "today, our legal team will redirect the engagement to your competitor.\n\n"
            "Please treat this as your highest priority. "
            "My direct line: +1-555-0192.\n\n"
            "Sarah Johnson\n"
            "VP of Procurement, BigClient Corp"
        ),
        "timestamp": "2024-01-15T10:05:00Z",
        "has_attachment": True,
        "true_priority": "urgent",
        "true_department": "Sales",
        "is_spam": False,
        "needs_response": True,
    },
    {
        "email_id": "E003",
        "sender": "security-desk@techcorp.com",
        "subject": "URGENT: Employee Workplace Accident – Immediate HR Action Required",
        "body": (
            "HR Team,\n\n"
            "Employee Marcus Chen (ID: EMP-4821) was involved in a workplace accident "
            "at the Chicago office 15 minutes ago and has been transported to Northwestern "
            "Memorial Hospital.\n\n"
            "Per company policy, we need IMMEDIATELY:\n"
            "  1. Incident report filing\n"
            "  2. Workers' compensation initiation\n"
            "  3. Emergency contact notification (wife: Linda Chen, 555-0847)\n"
            "  4. OSHA notification assessment\n\n"
            "Please respond IMMEDIATELY.\n\n"
            "Security Desk"
        ),
        "timestamp": "2024-01-15T11:30:00Z",
        "has_attachment": False,
        "true_priority": "urgent",
        "true_department": "HR",
        "is_spam": False,
        "needs_response": True,
    },
    {
        "email_id": "E004",
        "sender": "ciso@techcorp.com",
        "subject": "SECURITY BREACH DETECTED – Unauthorized Access to Customer PII",
        "body": (
            "All,\n\n"
            "Our IDS flagged a critical security breach at 11:47 AM today.\n\n"
            "Details:\n"
            "  - 3 unauthorized access attempts to the customer PII database SUCCEEDED\n"
            "  - Source IP: 185.220.101.x (known Tor exit node)\n"
            "  - ~2,300 customer records potentially exposed\n"
            "  - Breach window: 11:41 AM – 11:47 AM\n\n"
            "GDPR / CCPA notification requirements may be triggered. Legal must be looped in "
            "immediately. Security has isolated the affected systems but the breach must be "
            "forensically confirmed.\n\n"
            "Action required NOW.\n\n"
            "CISO"
        ),
        "timestamp": "2024-01-15T12:00:00Z",
        "has_attachment": True,
        "true_priority": "urgent",
        "true_department": "IT",
        "is_spam": False,
        "needs_response": True,
    },
    # NORMAL
    {
        "email_id": "E005",
        "sender": "vendor@supplierco.com",
        "subject": "Invoice #INV-2024-0892 for Q4 Software Licenses",
        "body": (
            "Dear Finance Team,\n\n"
            "Please find attached Invoice #INV-2024-0892 for software license renewals "
            "covering Q4 2024.\n\n"
            "  - Microsoft 365 Business Premium: 150 seats × $22/mo = $3,300\n"
            "  - Slack Pro: 150 seats × $7.25/mo = $1,087.50\n"
            "  - Total due: $4,387.50\n"
            "  - Payment due: February 15, 2024 (Net 30)\n\n"
            "Please process within standard payment terms.\n\n"
            "Best regards,\n"
            "SupplierCo Accounts Receivable"
        ),
        "timestamp": "2024-01-15T08:00:00Z",
        "has_attachment": True,
        "true_priority": "normal",
        "true_department": "Finance",
        "is_spam": False,
        "needs_response": False,
    },
    {
        "email_id": "E006",
        "sender": "james.wilson.new@techcorp.com",
        "subject": "New Employee Onboarding – Starting Monday, Need Logistics",
        "body": (
            "Hi HR Team,\n\n"
            "I'm James Wilson, joining as a Senior Software Engineer this coming Monday. "
            "I received my offer letter but have not yet received:\n\n"
            "  1. Laptop setup / shipping instructions\n"
            "  2. Badge / office access information\n"
            "  3. First-week schedule and team contacts\n\n"
            "Could you please send these details before the weekend so I can prepare?\n\n"
            "Looking forward to joining!\n\n"
            "James Wilson"
        ),
        "timestamp": "2024-01-15T08:45:00Z",
        "has_attachment": False,
        "true_priority": "normal",
        "true_department": "HR",
        "is_spam": False,
        "needs_response": True,
    },
    {
        "email_id": "E007",
        "sender": "compliance@techcorp.com",
        "subject": "Annual Data Privacy Training – All Staff Completion Required by Jan 31",
        "body": (
            "All Staff,\n\n"
            "As part of our compliance requirements, every employee must complete the "
            "'Data Privacy & GDPR Fundamentals 2024' training by January 31, 2024.\n\n"
            "  - Duration: ~45 minutes\n"
            "  - Link: training.company.com/privacy-2024\n"
            "  - Deadline: January 31, 2024\n\n"
            "Non-completion will be escalated to department managers and noted in performance reviews.\n\n"
            "Compliance Team"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": False,
        "true_priority": "normal",
        "true_department": "HR",
        "is_spam": False,
        "needs_response": False,
    },
    {
        "email_id": "E008",
        "sender": "emma.rodriguez@startup.io",
        "subject": "Enterprise Partnership Inquiry – 200-Seat Deal",
        "body": (
            "Hi Sales Team,\n\n"
            "We are a Series B startup ($50 M raised, 200 employees) looking for an "
            "enterprise software solution. After evaluating several vendors we are most "
            "interested in yours and would like to discuss:\n\n"
            "  1. Enterprise pricing for 200 seats\n"
            "  2. SSO / SAML integration\n"
            "  3. Custom SLA options\n"
            "  4. Implementation and onboarding support\n\n"
            "Can someone schedule a demo this week?\n\n"
            "Best,\n"
            "Emma Rodriguez\n"
            "Head of Operations, StartupIO"
        ),
        "timestamp": "2024-01-15T10:30:00Z",
        "has_attachment": False,
        "true_priority": "normal",
        "true_department": "Sales",
        "is_spam": False,
        "needs_response": True,
    },
    {
        "email_id": "E009",
        "sender": "tom.k@gmail.com",
        "subject": "Billing Error – Charged Twice for Monthly Subscription",
        "body": (
            "Hello Support,\n\n"
            "I noticed I was charged twice on my credit card for this month's subscription.\n\n"
            "  - Transaction 1: Jan 5, 2024 – $29.99 (legitimate)\n"
            "  - Transaction 2: Jan 5, 2024 – $29.99 (duplicate)\n\n"
            "My account email: tom.k@gmail.com\n"
            "Order reference: ORD-2024-88721\n\n"
            "Please refund the duplicate charge. I have been a customer for three years and "
            "this has never happened before. Happy to provide card statement screenshot if needed.\n\n"
            "Thanks,\n"
            "Tom"
        ),
        "timestamp": "2024-01-15T11:00:00Z",
        "has_attachment": True,
        "true_priority": "normal",
        "true_department": "Support",
        "is_spam": False,
        "needs_response": True,
    },
    {
        "email_id": "E010",
        "sender": "legal@partnerco.com",
        "subject": "NDA Review Required Before Partnership Call Next Week",
        "body": (
            "Dear Legal Team,\n\n"
            "We have a partnership discovery call scheduled for January 22nd. "
            "Before the call, both parties need to sign a mutual NDA.\n\n"
            "Attached is our standard mutual NDA template. Please review and either "
            "sign as-is or return with redlines by January 20th so we have time "
            "to align before the call.\n\n"
            "Please let us know if you need our legal counsel's contact details.\n\n"
            "Best,\n"
            "PartnerCo Legal"
        ),
        "timestamp": "2024-01-15T13:00:00Z",
        "has_attachment": True,
        "true_priority": "normal",
        "true_department": "Legal",
        "is_spam": False,
        "needs_response": True,
    },
    # LOW PRIORITY
    {
        "email_id": "E011",
        "sender": "newsletter@industryblog.com",
        "subject": "Tech Industry Weekly: AI Trends, Cloud Updates & More",
        "body": (
            "Tech Industry Weekly Newsletter\n\n"
            "This week's highlights:\n"
            "  • AI adoption in enterprise grows 34 % YoY\n"
            "  • AWS announces new Southeast Asia regions\n"
            "  • Kubernetes 1.29 released with new features\n"
            "  • Top 10 DevOps tools of 2024\n\n"
            "Read more at: industryblog.com/weekly\n\n"
            "To unsubscribe, click here.\n\n"
            "Industry Blog Team"
        ),
        "timestamp": "2024-01-15T07:00:00Z",
        "has_attachment": False,
        "true_priority": "low",
        "true_department": "IT",
        "is_spam": False,
        "needs_response": False,
    },
    {
        "email_id": "E012",
        "sender": "facilities@techcorp.com",
        "subject": "Monthly Office Supplies Order – Submit Requests by Friday",
        "body": (
            "Team,\n\n"
            "It's time for our monthly office supplies order. "
            "If you need anything for your workspace, please submit requests by Friday EOD.\n\n"
            "Common items: pens, notebooks, sticky notes, printer paper, etc.\n\n"
            "Submit here: facilities.company.com/supplies\n\n"
            "Facilities Team"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": False,
        "true_priority": "low",
        "true_department": "HR",
        "is_spam": False,
        "needs_response": False,
    },
    {
        "email_id": "E013",
        "sender": "alex.turner@techcorp.com",
        "subject": "Team Lunch Suggestion for Friday – Bella Roma?",
        "body": (
            "Hey everyone!\n\n"
            "Just wanted to float the idea of a team lunch this Friday at noon. "
            "I was thinking we could try that new Italian place on 5th Street – Bella Roma. "
            "Great lunch menu and only a 5-minute walk from the office.\n\n"
            "Let me know if you're interested and I'll make a reservation!\n\n"
            "Cheers,\n"
            "Alex"
        ),
        "timestamp": "2024-01-15T09:30:00Z",
        "has_attachment": False,
        "true_priority": "low",
        "true_department": "HR",
        "is_spam": False,
        "needs_response": False,
    },
    # SPAM
    {
        "email_id": "E014",
        "sender": "noreply@prize-winner-2024.xyz",
        "subject": "CONGRATULATIONS!!! You Won $500,000 – Claim NOW!!!",
        "body": (
            "CONGRATULATIONS!!!\n\n"
            "You have been SELECTED as our GRAND PRIZE WINNER of $500,000 USD!!!\n\n"
            "To claim your prize you MUST:\n"
            "  1. Provide your full name, home address, and bank account details\n"
            "  2. Pay a $199 processing fee via wire transfer\n"
            "  3. Respond within 24 hours or forfeit your prize\n\n"
            "Send your details to: claim@prize-winner-2024.xyz\n\n"
            "This is NOT spam! Legitimate notification from the International Lottery Commission.\n\n"
            "CLAIM YOUR PRIZE NOW!!!!!"
        ),
        "timestamp": "2024-01-15T08:15:00Z",
        "has_attachment": False,
        "true_priority": "low",
        "true_department": None,
        "is_spam": True,
        "needs_response": False,
    },
    {
        "email_id": "E015",
        "sender": "security-alert@paypa1-secure.net",
        "subject": "Your PayPal Account Has Been Compromised – Verify Immediately",
        "body": (
            "Dear Valued Customer,\n\n"
            "We detected suspicious activity. Your account has been TEMPORARILY SUSPENDED.\n\n"
            "To restore access click: http://paypa1-secure.net/verify?token=abc123\n\n"
            "You MUST verify within 2 hours or your account will be PERMANENTLY DELETED.\n\n"
            "Required verification:\n"
            "  - Full name & date of birth\n"
            "  - Social Security Number\n"
            "  - Credit card number and CVV\n\n"
            "PayPal Security Team"
        ),
        "timestamp": "2024-01-15T07:30:00Z",
        "has_attachment": False,
        "true_priority": "low",
        "true_department": None,
        "is_spam": True,
        "needs_response": False,
    },
]

#   Task email selections 
# Task 1 – Easy: 5 emails, varied priorities, no spam
TASK_EMAILS: Dict[str, List[str]] = {
    "priority_sort": ["E001", "E002", "E005", "E011", "E013"],
    #  expected: urgent, urgent, normal, low, low

    # Task 2 – Medium: 8 emails, varied priority + departments, no spam
    "department_routing": [
        "E001", "E002", "E003", "E005", "E006", "E008", "E009", "E010"
    ],
    # expected depts: IT, Sales, HR, Finance, HR, Sales, Support, Legal

    # Task 3 – Hard: 10 emails including 2 spam, need response drafts for urgent
    "full_triage": [
        "E001", "E002", "E003", "E005", "E006", "E007", "E008", "E009", "E014", "E015"
    ],
    # 3 urgent + 5 normal + 2 spam
}

EMAIL_MAP: Dict[str, Dict[str, Any]] = {e["email_id"]: e for e in EMAILS}
