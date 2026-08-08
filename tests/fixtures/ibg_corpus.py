"""Ground-truth corpus for IBG / interbank receipt extraction.

Every sample is REAL receipt text: either pulled from this repo (`data/uploads/`,
`logs/`, `debug_failures/`) or transcribed from receipts supplied by the owner.
Legal disclaimers and marketing footers are trimmed; everything that affects
extraction -- labels, values, column order, page footers -- is preserved
verbatim, because the layout *is* the difficulty.

`expected` values were read off the source documents by hand.
`traps` lists values on each receipt that a naive rule grabs by mistake. A field
extractor is correct only if it returns `expected` AND never returns a trap.

Four things this corpus is designed to prove:

1. **Multiple references.** Most receipts carry a bank-issued primary reference,
   one or more bank-issued secondary references (service / PayNet / DuitNow /
   UETR / batch / channel), and a payer-supplied reference. All must be captured
   and labelled; only a `bank_primary` may become the scalar `reference_id`.
2. **Amount is not a reference.** `amount` (transferred), `fee` and
   `total_debit` are three different numbers, and none of them is a reference
   number that happened to sit near the word "Amount".
3. **Invoice line items are not the amount.** Several receipts carry an invoice
   table whose rows are money.
4. **Date order is bank-dependent.** Every Malaysian portal is day-first;
   Citibank CitiDirect is month-first.
"""

from app.ibg.contract import (
    ROLE_BANK_PRIMARY as PRIMARY,
    ROLE_BANK_SECONDARY as SECONDARY,
    ROLE_PAYER_SUPPLIED as PAYER,
)

# ===========================================================================
# Maybank
# ===========================================================================

# Maybank M2E "Outward ACH" payment advice. Label above value, one per line,
# with a print header repeating at every page break and the beneficiary bank
# name split across a page boundary.
S1_MAYBANK_M2E_ACH = """A. Transaction Details
B. Beneficiary Details

Payment
Debit From Account No.
510000000001 (MYR) ACME LOGISTICS
HOLDINGS SDN. BHD.
Payment Mode
Outward ACH
Destination Country
MY - MALAYSIA
Date and Time
03 Aug 2026 12:23:57 MY (UTC+08:00)
Reference No.
MYIG2608033918802
Customer Ref No.
ACME CONTROL
Debit Reference
-
Debit Description
OOL12612032422
Transaction Currency
MYR
Transaction Amount
In Transaction Currency
10.60
Value Date (dd-mm-yyyy)
03 Aug 2026 MY (UTC+08:00)
Beneficiary Account No.
21000000000001
Beneficiary Name
ACME CONTROL (M) SDN BHD
Resident
Yes
Beneficiary Bank Code
RHBAMYKL
Beneficiary Bank Name
Cash Management System
04/08/2026, 07:57
Maybank M2E
https://www.maybank2e.com/m2e/portal/tran.view?do=ReLoad
1/4
 -
-
-
RHB ISLAMIC BANK
Beneficiary Bank Address
-
Business Reg. No.
100001A
Police/Army ID/Passport No.
-
Send Payment Advice
No Advice
04/08/2026, 07:57
Maybank M2E
https://www.maybank2e.com/m2e/portal/tran.view?do=ReLoad
2/4
 Applicant ID/ Unit Code
100002B
Zip/Postal Code
41200
Transaction Status
Successfully Sent to Bank
User
Activities
Date and Time
AISYAH BT OMAR
Submit
03 Aug 2026 12:23:58 MY (UTC+08:00)
LIM WEI KEAT
Approve
03 Aug 2026 15:08:21 MY (UTC+08:00)
04/08/2026, 07:57
Maybank M2E
https://www.maybank2e.com/m2e/portal/tran.view?do=ReLoad
4/4
"""

# Same M2E form, truncated and column-scrambled: labels arrive as one block and
# values as another. There is NO reference number on this document.
S2_MAYBANK_M2E_TRUNCATED = """04/08/2026, 07:57 Maybank M2E
Cash Management System
Payment
A. Transaction Details
Debit From Account No. Transaction Currency
510000000001 (MYR) ACME LOGISTICS MYR
HOLDINGS SDN. BHD.
Transaction Amount
Payment Mode
Beneficiary Name
Destination Country
Value Date (dd/mm/yyyy)
12.60
MY (MALAYSIA)
02 Aug 2026
"""

# Maybank2u "Open Interbank" receipt (OCR). Two competing reference-shaped
# values: the bank's "Reference number" and the payer's "Recipient reference".
S3_MAYBANK2U_OPEN_INTERBANK = """Open Interbank
Status: Successful
Reference number: 7901990048
Transaction date: 31 Oct 2025 12:09:52
Amount: RM10.60
Beneficiary Name : ACME CONTROL (M) SDN BHD
Receiving Bank : RHB BANK
Beneficiary Account Number : 21000000000001
Recipient reference: MXC12511726684
Other payment details:
Note: This receipt is computer generated and no signature is required.
maybank2u.com
"""

# Maybank GIRO/DuitNow receipt, OCR column scramble: all labels run together,
# then all values, in matching order. OCR damage: 'Transter', 'Suscessiul'.
S4_MAYBANK_GIRO_DUITNOW = """Thank You Here's your receipt. Transaction Successful
GIRO/DuitNow Transter
  31 Oct 2025, 10:31:57   Reference ID Status Transfer To Account Number \
Recipient Bank. Transfer Type Transfer Mode Recipien! Reference Payment Details \
7901764116 Suscessiul ACME CONTROL (Mj SON BHD 2100 0000 0000 015 RHB BANK \
Funds Transfer DuitNow Transfer DEMURAGE BL-ASB/2025-26/141A   Amount RM 1,144.80
"""

# Maybank2u Biz GIRO/DuitNow receipt. Clean two-column layout.
S6_MAYBANK2U_BIZ_GIRO = """Thank You
Here's your receipt.
Transaction Successful
GIRO/DuitNow Transfer 06 Aug 2026, 17:03:08
Reference ID 9490679647
Status Successful
Transfer To ACME CONTROL (M) SDN BHD
Account Number 2100 0000 0000 01
Recipient Bank RHB BANK
Transfer Type Funds Transfer
Transfer Mode DuitNow Transfer
Recipient Reference SNK12612032540
Payment Details PGA2 - DETENTION
Amount RM 445.20
Malayan Banking Berhad (Co. Reg. : 199001000002)
Maybank2u Biz
"""

# Maybank M2E Outward ACH, second specimen. Beneficiary bank name is adjacent
# to its label here (unlike S1), and there are three payer-supplied refs.
S7_MAYBANK_M2E_ACH2 = """Completed
Task submitted
A. Transaction Details
B. Beneficiary Details
Payment
Debit From Account No.
510000000002 (MYR) NORTHWIND SHIPPING AGENCIES
SDN. BHD.
Payment Mode
Outward ACH
Destination Country
MY - MALAYSIA
Date and Time
06 Aug 2026 16:06:09 MY (UTC+08:00)
Reference No.
MYIG2608067512512
Customer Ref No.
IAL12612040140
Debit Reference
SA17963
Debit Description
IAL12612040140 - SA17963
Transaction Currency
MYR
Transaction Amount
In Transaction Currency
10.00
Value Date (dd-mm-yyyy)
06 Aug 2026 MY (UTC+08:00)
Beneficiary Account No.
21000000000001
Beneficiary Name
ACME CONTROL (M) SDN BHD
Resident
Yes
Beneficiary Bank Code
RHBAMYKL
Beneficiary Bank Name
RHB ISLAMIC BANK
Maybank Cash Management System
"""

# Maybank2E emailed Payment Advice. "Advice Reference no" is the bank primary;
# the beneficiary here is OCBC, not RHB.
S8_MAYBANK2E_ADVICE = """From: Maybank2E-admin@maybank.com.my
Sent: 28 July 2026 11:07
Subject: Payment Advice
Advice sending date : 28-Jul-2026 MY (UTC+08:00)
Advice Reference no : MYIG2607289982882
Page : 1/1
Applicant Details
Contact Name : EVERGREEN LOGISTICS SDN. BHD.
Telephone : 60 03 00000000 00
Beneficiary Details
Recipient Name : ADRIATIC SHIPPING COMPANY (M) S/B
Beneficiary Account No. : *******0002
Bank : OCBC BANK
Transaction Details
Transaction Type : Outward ACH
Customer Ref : -
Value Date : 28-Jul-2026 MY (UTC+08:00)
Payment Advice Amount : MYR 18,310.00
Purpose of Transfer : -
Advice Detail : PAY FOR DEM CHGS MYR 18,310.00
"""

# ===========================================================================
# Public Bank
# ===========================================================================

# PB enterprise "Interbank GIRO (IBG)" approval. Three references, three roles.
# Print stamp in the header is not the payment date.
S9_PBB_IBG_APPROVAL = """PBenterprise
Printed Date/Time: 7 Aug 2026 10:46:40
Reference No. : 2608070370908566
Service Reference No. : 559155
Transaction Approval
Transaction Details
Product Type
Reference No
Service Reference No
Approval Status
From Account
Transfer Mode
Favourite Name
Beneficiary Bank
Transfer Type
Beneficiary Name
Beneficiary Account No
Recipient Reference
Amount
Fee
Payment Date
Interbank GIRO (IBG)
2608070370908566
559155
Processed
3800000001 / SUMMIT LOGISTICS (M) SDN BHD
Favourite Account Transfer
AcMe CONTROL
RHB Islamic Bank Berhad
Fund Transfer
ACME CONTROL
21000000000001
DEM - TAWU4057398
MYR 954.00
MYR 0.00
07-Aug-2026
Copyright 2026 Public Bank Berhad 199001000001 (1000-A).
"""

# PB enterprise DuitNow slip. Four fee-shaped zeros surround the real amount.
S10_PBB_DUITNOW_SLIP = """Reference No. : 2607150361163858
Service Reference No. : 629362
PBenterprise
Printed Date/Time: 7 August 2026 10:57:26
Transaction Details
Product Type DuitNow Transfer
Reference No 2607150361163858
Service Reference No 629362
Approval Status Success
From Account 3800000002 / HARBOUR LOGISTICS SDN BHD
Transfer Mode Favourite Account Transfer
Favourite Name ACME CONTROL
Recipient's DuitNow ID Type Account Number
Recipient Bank RHB Bank Berhad
Transfer Type Fund Transfer
Recipient's DuitNow ID/Account No. 21000000000001/ACME CONTROL (M) SDN BHD
Recipient Reference ONE12612014237
Other Payment Details D D
Amount MYR 10.60
Fee MYR 0.00
Total Fee Charges MYR 0.00
SMS Fee MYR 0.00
Payment Date 15-Jul-2026
Copyright 2026 Public Bank Berhad (1000-A). ALL RIGHTS RESERVED
"""

# The same PB enterprise slip as a column-scrambled PDF text layer: label block
# then value block. This is the exact shape that makes the legacy engine return
# the Service Reference No (629362) as the amount.
S11_PBB_COLUMN_SCRAMBLE = """PBenterprise
Public Bank Berhad
Transaction Details
Product Type
Reference No
Service Reference No
Payment Date
Amount
Fee
DuitNow Transfer
2607150361163858
629362
15-Jul-2026
10.60
0.00
"""

# Public Bank fund-transfer notification email.
S12_PBB_NOTIFICATION_EMAIL = """Notification of Fund Transfer via PB enterprise from Public Bank Berhad
pbenterprise@publicbank.com.my
Dear Sir/Madam,
Please be informed that a fund transfer has been effected via PB enterprise:
SENDER NAME : PACIFIC NEPTUNE INTERNATIONAL SDN BHD
BENEFICIARY NAME : ACME CONTROL M SDN BHD
TRANSACTION DATE & TIME : 07 Aug 2026 08:12:49
TRANSACTION AMOUNT : MYR 21.20
BENEFICIARY BANK : RHB Islamic Bank Berhad
BENEFICIARY ACCOUNT NO. : 2*********0001
REFERENCE NO. : 2608060370480721
RECIPIENT REFERENCE : IAL12612040134
OTHER PAYMENT DETAILS : WHS12612040043
This notice is NOT a confirmation of payment.
"""

# ===========================================================================
# CIMB
# ===========================================================================

# CIMB BizChannel "Domestic Transfers / Interbank GIRO (IBG)" slip.
# Amount 106.00, fee 0.10, total debit 106.10 -- three distinct numbers.
S13_CIMB_BIZCHANNEL_IBG = """Domestic Transfers
Transfer Information
Transaction Status : Executed Successfully
Reference No. : 202608070343897254
Source Account : 8000000001 / VERTEX LOGISTICS MALAYSIA SDN. BHD. (MYR)
Services : Interbank GIRO (IBG)
Beneficiary Bank Information
Bank Name : RHB BANK BHD
Beneficiary Account Information
Account No : 21000000000001
Name : ACME CONTROL (M) SDN BHD
Validate Beneficiary ID : No
Additional Information
Amount : MYR 106.00
Transaction Fee : MYR 0.10
Tax Rate : 0.00%
Tax Amount : MYR 0.00
Total Debit Amount : MYR 106.10
Recipient's Reference : OOL12612038565
Other Payment Details :
Instruction Mode
Today : 07-Aug-2026
cimb.bizchannel.com.my
"""

# CIMB BizChannel transaction-notification email. "Domestic Transfers - IBG".
S14_CIMB_EMAIL_IBG = """From: mybusinesscare@cimb.com
Sent: Wednesday, August 5, 2026 7:02 PM
Subject: Transaction Notification : Success
We are pleased to inform you that a request to transfer into your account has
been submitted to CIMB Bank for processing.
Please refer to the payment details below:
Date/Time : 05-08-2026 19:01:59
Reference No. : 202608050343665310
Transaction Type : Domestic Transfers - IBG
From Account Name : ROSEWOOD SDN. BHD.
To Account : **********0001
Amount : MYR 31.80
Remark : SNK22610700234
Status : Success
BizChannel@CIMB Team
"""

# ===========================================================================
# RHB
# ===========================================================================

# RHB Reflex third-party transfer. Business Registration Number 100001A sits
# right beside the money block.
S15_RHB_REFLEX = """Payment - Transaction Status
3rd Party Account Transfer Information
Reference Number
260806446749
From Account
20000000000001 / NOVATECH
To Account
21000000000001 / ACME CONTROL (M) SDN. BHD.
Applicant Resident Status
Resident
Amount
RM 74.20
Fund Transfer Charges
RM 0.00
Service Tax Charges
RM 0.00
Total Amount to Debit
RM 74.20
IC / ID Number
Business Registration Number
100001A
Transfer Type
Post Dated Transfer
Payment Date
07-08-2026
Recipient`s Reference
NOVATECH BHD
Other Payment Details
HAL22610717193
Status
Processed
https://reflex.rhbgroup.com/rhbcams/corporate/login.jsp
"""

# RHB interbank advice, degraded OCR. Not much is recoverable -- the bar here is
# that the extractor does not invent values.
S5_RHB_DEGRADED = """      Applicani Dstails KITA-KAIUN    Beneficiary Details   \
ACME CONTROL SDK. BHD. I,
RHB BANK (RHBBMYKL)
Transaction Detalis
   WHS12511799 AYSIA) SDN BHD.     Aree MYR 10.69
"""

# ===========================================================================
# OCBC
# ===========================================================================

# OCBC Velocity DuitNow. Four references including a PayNet clearing reference.
S16_OCBC_VELOCITY = """Transfer to a bank locally
DuitNow
Transaction Status: Successful
Creation Date: 06 Aug 2026 10:52:29
From
Your Account
7300000001-MYR APEX FREIGHT (MALAYSIA) SDN. BHD.
Value Date
06 Aug 2026
Amount
53.00 MYR
Same day payment
To
Transfer Type
Fund Transfer
Beneficiary Bank Name
RHB BANK BERHAD
Beneficiary Account No.
21000000000001
Beneficiary Name
ACME CONTROL (M) SDN BHD
Recipient's Reference
SNK22610716537
Other Payment Details
PGPY-69353 I2607427
PayNet Reference No.
20260806OCBCMYKL010OCB26583413
Other Details (optional)
Your Reference No.
PGPY-69353
https://velocity.ocbc.com/digital/web/my/bfo-t2/ift/fund-transfer/duitnow
"""

# OCBC "GIRO Payment Details Report". Note the amount prints as "MYR - 752.60"
# with a stray hyphen between currency and value.
S17_OCBC_GIRO_REPORT = """GIRO Payment Details Report
Search Criteria
Report Type Transaction Report Name GIRO Details Report
Organisation ID NCABLE Country Malaysia
Account No. 7300000002 Date Range 07 Aug 2026 To 07 Aug 2026
OCBC Reference No. MYGP260806823259
Total Amount MYR - 752.60 Value Date 07 Aug 2026
Your Reference No MYGP200414483476 OCBC Reference No. MYGP260806823259
Source ACME CONTROL
Your Account No. 7300000002 - MYR NORTHERN CABLE SDN BHD
Item No. 1
Amount 752.60 MYR Require ID Checking N
Payee Name ACME CONTROL (M) SDN BH
Payee Account No. 21000000000001 Payee ID No.
Payee Bank Name RHB BANK KUALA LUMPUR
Recipient's Reference No. INV:CSE 22610716942
Item Status Pending Clearance
** End of Report **
Printed By NURUL BINTI HASSAN Page 1 of 1
Printed On 07 Aug 2026 12:59:04
"""

# ===========================================================================
# Hong Leong
# ===========================================================================

# Hong Leong ConnectFirst DEBIT ADVICE. Four references; two zero-value fee
# siblings sit directly under the amount.
S18_HONGLEONG_DEBIT_ADVICE = """HongLeong Bank    HongLeong Islamic Bank
connectFirst
DEBIT ADVICE
Date : 07-08-2026
Account No. : XXXXXXX0005
Account Name : CEDAR WAREHOUSING SDN
Your Account has been debited for the following transaction
Transaction Details
Transaction Reference No. : C765070826120713
Beneficiary Name : ACME CONTROL (M) SDN BHD
Payment Type : DuitNow Payment
DuitNow ID Type : Account No.
Value Date : 07-08-2026
DuitNow ID : RHBBMYKL - XXXXXXXXXX0001
Amount (MYR) : 445.20
DuitNow Reference No. : 20260807HLBBMYKL010OCB65523848
Service Charge (MYR) : 0.00
Channel Reference No. : CMSPCT26080700011638
GST Rate - Service Charge (%) : 0
Recipient Reference : DET-WHSU6510076
GST Amount-Service Charge (MYR) : 0.00
Other Details : -
"""

# Hong Leong ConnectFirst CREDIT ADVICE -- received by the beneficiary. The
# issuing bank is Hong Leong (the advising bank); the payer banks elsewhere.
S19_HONGLEONG_CREDIT_ADVICE = """HongLeong Bank    HongLeong Islamic Bank
connectFirst
CREDIT ADVICE
Dear Sir/Madam,
On request of our customer, we have effected payment to your Account
as per the details below:
Date : 07/08/2026
Ordering Customer Account No. : XXXXXXX0004
Ordering Customer Account Name : BRIGHTPATH SDN BHD
Transaction Details
Transaction Reference No. : C773070826095112
Payment Type : Payment to 3rd Party Account - IBG
Value Date : 07-08-2026
Beneficiary Name : ACME CONTROL (M) SDN BHD
Amount ( MYR ) : 10.60
Beneficiary Account No. : XXXXXXXXXX0001
Beneficiary Bank : RHBB,RHB BANK BHD
Bank Reference No. : IBGCMPRHBB2608070001189
Recipient Reference : INV NO:12036941
Other Details : IA0060726
"""

# ===========================================================================
# UOB
# ===========================================================================

# UOB "Interbank GIRO". Bank Reference is the primary; the recipient reference
# is a run of loose digits that must not be mistaken for money or an account.
S20_UOB_INTERBANK_GIRO = """UOB
Pending Authorise
Interbank GIRO
Bank Reference
FT26080178167822
Application Date: 07/08/2026
From
Company Name
ALPHA LOGISTICS SDN BHD
Account Name
Current Account
Account Number
2000000001
Currency
MYR
To
Payee Name
ACME CONTROL SDN BHD
Bank Name
RHB BANK
Bank Code
100002186
Account Number
21000000000001
Payee Residence Status
Resident
Amount & When
Value Date
07/08/2026
Transaction Amount
42.40
Transaction Currency
MYR
Recipient Reference
iv 10710913 10707642
Other Payment Details
10707097 10706703
Date of Export: 07/08/2026 | Time of Export: 12:31:05
"""

# ===========================================================================
# AmBank
# ===========================================================================

# AmBank DuitNow payment slip. Debit and credit amounts are equal.
S21_AMBANK_DUITNOW = """AmBank    AmBank Islamic
Page 1 of 1
Recipient Ref./Customer Ref. No. Transaction Status Transfer Type
DEPOSIT PAYMENT Pending for Approval DUITNOW | DuitNow Transfer
Transaction Reference Number Transaction Status Reason Transaction Date
FIN0708261152699 - 07/08/2026
Transaction Details
Debit Account Credit Account Credit Bank Name
8880000000001 | LUMEN LIGHTING
SDN. BHD.
21000000000001 RHB | RHB BANK BERHAD
Debit Amount Credit Amount Value Date
MYR 6,010.60 MYR 6,010.60 07/08/2026
Batch Ref Number
0708202611030843151
Channel
Internet Portal
Payee/Beneficiary Details
Payee Name Payee Residency Status Payment Type
ACME CONTROL (M) SDN BHD Resident Fund Transfer
Generated on 07 August 2026 at 11:03 AM by PRIYA
"""

# AmBank Group DuitNow Transfer Advice.
S22_AMBANK_TRANSFER_ADVICE = """AmBank Group
BLUE WAVE SHIPPING LINES SDN.BHD.
DuitNow Transfer
Transfer Advice
Reference Number: DTF20260803013679
Transaction Date & Time: 03/08/2026 16:27:28
From Account Number: 8880000000002
Recipient Bank: RHB BANK
Recipient Account Number: 21000000000001
Recipient Name: ACME CONTROL (M) SDN BHD
Transfer Date: 03/08/2026
Recipient's Reference: MULTIPLE INVOICE
Other Payment Details: -
Transaction Status: Processed
Transfer Amount: MYR 12,587.50
Transfer Fee: MYR 0.00
Total Debit: MYR 12,587.50
Copyright AmBank (M) Berhad (1001-B) Computer generated : 03/08/2026 16:27:57
"""

# ===========================================================================
# Alliance
# ===========================================================================

# Alliance BizSmart DuitNow. Label block then value block.
S23_ALLIANCE_BIZSMART = """View Successful / Failed - Detail
DuitNow (Pay to Account / Instant Transfer)
Reference No :
Transaction Status :
Transaction Date, Time :
From Account :
To Account :
Beneficiary Name :
Beneficiary Bank :
Recipient's Reference :
Other Payment Details :
Payment Type :
Payment Date :
Amount (MYR) :
Service Charge(MYR) :
Total Amount(MYR) :
AOBIFT07082026075791
Successful
07/08/2026 11:49:07
070000000000001 - ORBIT GLOBAL LOGISTICS SDN. BHD. ( MYR )
21000000000001
ACME CONTROL (M) SDN BHD
RHB BANK BHD
OOL22610716770
2330685513
Fund Transfer
07/08/2026
10.00
0.00
10.00
https://www.alliancebizsmart.com.my/business/
"""

# ===========================================================================
# HSBC
# ===========================================================================

# HSBC Payment Advice. The invoice row prints "MYR -31.800" -- negative sign and
# three decimals. The remittance amount is the answer.
S24_HSBC_PAYMENT_ADVICE = """HSBC
Payment Advice
Advice sending date: 07 Aug 2026
Advice reference no: A2u38OFF1OT1
Page: 1/1
Transaction type: ACH credits
Beneficiary's name: D ANACME CONTROL (M)
Beneficiary's bank: RHB BANK BERHAD
Malaysia
Beneficiary's account: 210000000*****
Customer reference: 2536
2nd party reference: D and D Cont
Remittance amount: MYR31.80
Value date: 06 Aug 2026
Remitter's name: SEALINE CARGO SYS ( M )
Remitting bank: HSBC Bank Malaysia Berhad
Instruction reference: 9002490LZ2KO
Remitter to beneficiary information:
/EREF/MED12612037415
MED12612037415
Document Number Amount Date Reference Type
MED12612037415 MYR -31.800 04/08/2026 006718
Issued by HSBC Bank Malaysia Berhad
"""

# ===========================================================================
# Standard Chartered
# ===========================================================================

# Standard Chartered Straight2Bank. Five bank-issued references, and an invoice
# table whose four rows are all money.
S25_SCB_STRAIGHT2BANK = """standard chartered
Payment Transaction Details
Generated on: Aug 07, 2026 10:50 AM SGT
Generated by: IRA
Payment Identification
End to End ID (Your Reference) 00117518
Bank Payment Reference Q0483528
UETR Number e14ae93a-0499-4739-8ae9-3a0499173989
Back Office Reference -
Import Reference 26080710nf880730729802000
Batch Reference C0166905
Status Batched for Authorisation
Debtor Details
Debit Account GLOBEX FORWARDING (MALAYSIA) S - 700000000001 - SCBLMYKXXXX - MY - MYR
Payment Details
Debit Amount MYR 241.20
Credit Amount MYR 241.20
Debit Date 07/08/2026
Value Date 07/08/2026
Payment Type Automated Clearing House (ACH)
Beneficiary Details
Name ACME CONTROL (M) SDN BHD
ACCOUNT 21000000000001
Beneficiary Bank
Bank Code Type SWIFT
Bank Code RHBAMYKLXXX
Bank Name RHBAMYKLXXX
Invoice Details
Invoice Type 4-Column Invoice
Reference Description Invoice Date Invoice Amount
CSE126DNDI26 - 03/08/2026 120.00
MGL126120357 - 03/08/2026 100.00
ONE126120081 - 09/07/2026 10.60
RCL126120098 - 10/07/2026 10.60
Total Amount MYR 241.20
MYR Two Hundred Forty-One ringgit and Two sen
"""

# ===========================================================================
# Citibank -- the month-first outlier
# ===========================================================================

# Citi CitiDirect ACH Credit/GIRO. "Value Date 08/05/2026" is 5 August 2026,
# NOT 8 May: CitiDirect prints US month-first. Every other portal in this
# corpus is day-first, which is exactly why this sample exists.
S26_CITIBANK_CITIDIRECT = """citi
CitiDirect
Payment To: Pacific Container Line Ltd
Summary
Beneficiary Name
Pacific Container Line Ltd
Account Name
NEXUS TRANSPORT MALAYSIA SDN BHD
Transaction Reference Number
C50026080500004
Beneficiary Account Number
200000000001
Debit Account
110000001
Tracking Number(UETR)
db1e687ba63e453f947d2c2b72bf2936
Status
Processed
Payment Amount
120.00
Value Date
08/05/2026
Payment Details
Payment Method
ACH Credit/GIRO
Created Date & Time
08/05/2026 11:48:25 AM GMT+08:00
Payment Type
PAYLINK GIRO, Malaysia
Payment Details
4124014527
Payment Currency
MYR - MALAYSIAN RINGGIT
Debit Account Details
Branch Name
CITIBANK BERHAD 1002-C (MALAYSIA)
Beneficiary Details
Beneficiary Bank Routing Method
INTERBANK GIRO
Beneficiary Bank Routing Code
HBMBMYKL
Beneficiary Bank Name
HSBC BANK MALAYSIA BERHAD
Printed on 08/05/2026 12:19:22 (GMT+08:00) Page 1 of 2
"""

# ===========================================================================
# MUFG
# ===========================================================================

# MUFG GCMS Plus "Domestic - Interbank GIRO". The amount is a group total; the
# invoice list below it has nine money rows that must not win. Value Date is
# dotted ISO (2026.08.03).
S27_MUFG_GCMS_IBG = """MUFG COMSUITE GCMS Plus
Transaction Details TR2004S0
Payment Type and Settlement Account Information
Payment Type Domestic - Interbank GIRO
Group No. G-GCP1803520
Entry No. 3644-GCP3705323
Entry Type New
Status Approved (Submitted to Bank)
Account MYR 602000 CURRENT DEPOSIT
Bank Name MUFG Bank (Malaysia) Berhad
Branch Name Kuala Lumpur
Account Name
SAKURA PACKING (MALAYSIA) SDN BHD
Group Information
Value Date 2026.08.03
Debit Type Single Debit / Consolidated
Transaction Details
Currency MYR
Amount 10,982.12
Segment Code Third Party Transfer
Beneficiary Details
Name ACME CONTROL
Account No. 21000000000001
Beneficiary Bank Information
Bank Code / National Clearing Code
10000218
RHB BANK
Notification Details
Recipient's Reference P79153
Invoice List
# Invoice Date Invoice No. Invoice Amount
1 2026.07.31 KMT12612032470 1,817.90
2 2026.07.31 KMT12612032510 1,817.90
3 2026.07.31 KMT12612032511 1,817.90
4 2026.07.31 KMT12612032516 1,817.90
5 2026.07.31 KMT12612032601 82.08
6 2026.07.21 KMT12612021474 84.24
7 2026.07.29 KMT12612030199 1,627.10
8 2026.07.29 KMT12612030207 1,627.10
9 2026.07.29 ONE126600130018135 290.00
Total Amount / Number of Invoices
Currency Total Invoice Amount Total Number of Invoices
MYR 10,982.12 9
https://e.ebusiness.bk.mufg.jp/pfa2/a001/gp/WTR200405_MN.do
"""

# ===========================================================================
# Bank of China
# ===========================================================================

# Bank of China iGTB DuitNow. Three zero-value charge lines.
S28_BANKOFCHINA_IGTB = """BANK OF CHINA
iGTB
Transaction Details
DuitNow
Pending 1st Authorisation
iGTB Reference
40002132227
Transaction Date and Time
06 Aug 2026 17:47 GMT+08:00
Maker
EMY007
From
Debit Account*
ENSIGN LOGISTICS SDN BHD
100000000000001
MALAYSIA | BOC Group Account
Charges Option *
OUR
To
Credit Account*
Beneficiary Account Number
ACME CONTROL (M) SDN BHD
21000000000001
Current Account
Payment Amount MYR 50.00
Beneficiary Bank/Institution Information*
RHB Bank Berhad
Transaction Detail
Amount
Debit Amount (Base)* MYR 50.00
Payment Amount (Equivalent)* MYR 50.00
Charges
Total Charges MYR 0.00
Remittance Commission MYR 0.00
Tax Amount MYR 0.00
Supplementary Information
Recipient's Reference*
EMC12612031446,2032331, YML12612033315
Customer Reference
YBP26080022
"""


CORPUS = [
    # ---- Maybank ----------------------------------------------------------
    {
        "id": "S1_MAYBANK_M2E_ACH",
        "text": S1_MAYBANK_M2E_ACH,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "MYIG2608033918802",
            "references": [
                ("MYIG2608033918802", PRIMARY),
                ("ACME CONTROL", PAYER),          # Customer Ref No.
                ("OOL12612032422", PAYER),       # Debit Description
            ],
            "bank_name": "Maybank",
            "beneficiary_bank": "RHB Islamic Bank",
            "transaction_date": "2026-08-03",
            "amount": "10.60",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["510000000001", "21000000000001", "100001A",
                             "100002B", "RHBAMYKL"],
            "transaction_date": ["04/08/2026", "2026-08-04"],
            "amount": ["41200", "510000000001", "21000000000001"],
            "bank_name": ["RHB Islamic Bank", "RHB", "Unknown", "Generic"],
        },
        "notes": "Beneficiary bank name split from its label by a page footer.",
    },
    {
        "id": "S2_MAYBANK_M2E_TRUNCATED",
        "text": S2_MAYBANK_M2E_TRUNCATED,
        "ocr_used": False,
        "is_ibg": False,
        "expected": {
            "reference_id": None,
            "references": [],
            "bank_name": "Maybank",
            "transaction_date": "2026-08-02",
            "amount": "12.60",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["510000000001", "12.60", "02 Aug 2026"],
            "transaction_date": ["04/08/2026", "2026-08-04"],
            "amount": ["510000000001"],
            "bank_name": ["Unknown", "Generic"],
        },
        "notes": "No reference on the document. None is the correct answer.",
    },
    {
        "id": "S3_MAYBANK2U_OPEN_INTERBANK",
        "text": S3_MAYBANK2U_OPEN_INTERBANK,
        "ocr_used": True,
        "is_ibg": True,
        "expected": {
            "reference_id": "7901990048",
            "references": [
                ("7901990048", PRIMARY),
                ("MXC12511726684", PAYER),
            ],
            "bank_name": "Maybank",
            "transaction_date": "2025-10-31",
            "amount": "10.60",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["MXC12511726684", "21000000000001"],
            "amount": ["21000000000001"],
            "bank_name": ["RHB Bank", "RHB", "Unknown", "Generic"],
        },
    },
    {
        "id": "S4_MAYBANK_GIRO_DUITNOW",
        "text": S4_MAYBANK_GIRO_DUITNOW,
        "ocr_used": True,
        "is_ibg": True,
        "expected": {
            "reference_id": "7901764116",
            "references": [("7901764116", PRIMARY)],
            # Not recoverable: "Maybank" appears nowhere in the body.
            "bank_name": None,
            "transaction_date": "2025-10-31",
            "amount": "1144.80",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["2100 0000 0000 015", "210000000000015",
                             "BL-ASB/2025-26/141A"],
            "amount": ["2124", "6660", "0013", "435"],
            "bank_name": ["RHB Bank", "RHB", "DuitNow", "GIRO", "Unknown",
                          "Generic"],
        },
        "notes": "Label block then value block. OCR damage throughout.",
    },
    {
        "id": "S6_MAYBANK2U_BIZ_GIRO",
        "text": S6_MAYBANK2U_BIZ_GIRO,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "9490679647",
            "references": [
                ("9490679647", PRIMARY),
                ("SNK12612032540", PAYER),
            ],
            "bank_name": "Maybank",
            "transaction_date": "2026-08-06",
            "amount": "445.20",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["2100 0000 0000 01", "212466600 1343"],
            "amount": ["2124", "6660", "0013", "199001000002"],
            "bank_name": ["RHB Bank", "RHB", "DuitNow", "Unknown", "Generic"],
        },
    },
    {
        "id": "S7_MAYBANK_M2E_ACH2",
        "text": S7_MAYBANK_M2E_ACH2,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "MYIG2608067512512",
            "references": [
                ("MYIG2608067512512", PRIMARY),
                ("IAL12612040140", PAYER),
                ("SA17963", PAYER),
            ],
            "bank_name": "Maybank",
            "beneficiary_bank": "RHB Islamic Bank",
            "transaction_date": "2026-08-06",
            "amount": "10.00",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["510000000002", "21000000000001", "RHBAMYKL"],
            "amount": ["510000000002", "21000000000001"],
            "bank_name": ["RHB Islamic Bank", "RHB", "Unknown", "Generic"],
        },
    },
    {
        "id": "S8_MAYBANK2E_ADVICE",
        "text": S8_MAYBANK2E_ADVICE,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "MYIG2607289982882",
            "references": [("MYIG2607289982882", PRIMARY)],
            "bank_name": "Maybank",
            "beneficiary_bank": "OCBC",
            "transaction_date": "2026-07-28",
            "amount": "18310.00",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["60 03 00000000 00", "2300"],
            "amount": ["2300", "60033167121302"],
            "bank_name": ["OCBC", "Unknown", "Generic"],
        },
        "notes": "Beneficiary is OCBC, not RHB. Issuer is Maybank.",
    },

    # ---- Public Bank ------------------------------------------------------
    {
        "id": "S9_PBB_IBG_APPROVAL",
        "text": S9_PBB_IBG_APPROVAL,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "2608070370908566",
            "references": [
                ("2608070370908566", PRIMARY),
                ("559155", SECONDARY),
                ("DEM - TAWU4057398", PAYER),
            ],
            "bank_name": "Public Bank",
            "beneficiary_bank": "RHB Islamic Bank",
            "transaction_date": "2026-08-07",
            "amount": "954.00",
            "fee": "0.00",
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["3800000001", "21000000000001", "199001000001"],
            "transaction_date": ["2026-08-07T10:46:40", "10:46:40"],
            "amount": ["559155", "3800000001", "21000000000001",
                       "199001000001", "0.00"],
            "bank_name": ["RHB Islamic Bank", "RHB", "Unknown", "Generic"],
        },
        "notes": "Product Type is literally 'Interbank GIRO (IBG)'.",
    },
    {
        "id": "S10_PBB_DUITNOW_SLIP",
        "text": S10_PBB_DUITNOW_SLIP,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "2607150361163858",
            "references": [
                ("2607150361163858", PRIMARY),
                ("629362", SECONDARY),
                ("ONE12612014237", PAYER),
            ],
            "bank_name": "Public Bank",
            "transaction_date": "2026-07-15",
            "amount": "10.60",
            "fee": "0.00",
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["3800000002", "21000000000001"],
            # The print stamp is 7 Aug; the payment date is 15 Jul.
            "transaction_date": ["2026-08-07"],
            "amount": ["629362", "3800000002", "0.00"],
            "bank_name": ["RHB Bank", "RHB", "DuitNow", "Unknown", "Generic"],
        },
        "notes": "Three separate zero-value fee lines around the real amount.",
    },
    {
        "id": "S11_PBB_COLUMN_SCRAMBLE",
        "text": S11_PBB_COLUMN_SCRAMBLE,
        "ocr_used": False,
        "is_ibg": False,
        "expected": {
            "reference_id": "2607150361163858",
            "references": [
                ("2607150361163858", PRIMARY),
                ("629362", SECONDARY),
            ],
            "bank_name": "Public Bank",
            "transaction_date": "2026-07-15",
            "amount": "10.60",
            "fee": "0.00",
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["10.60", "0.00"],
            "amount": ["629362", "2607150361163858"],
            "bank_name": ["Unknown", "Generic"],
        },
        "notes": "REGRESSION GUARD for the reported bug: the legacy engine "
                 "returns 629362 (the Service Reference No) as the amount.",
    },
    {
        "id": "S12_PBB_NOTIFICATION_EMAIL",
        "text": S12_PBB_NOTIFICATION_EMAIL,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "2608060370480721",
            "references": [
                ("2608060370480721", PRIMARY),
                ("IAL12612040134", PAYER),
                ("WHS12612040043", PAYER),
            ],
            "bank_name": "Public Bank",
            "beneficiary_bank": "RHB Islamic Bank",
            "transaction_date": "2026-08-07",
            "amount": "21.20",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["21000000000001"],
            "amount": ["2608060370480721"],
            "bank_name": ["RHB Islamic Bank", "RHB", "Unknown", "Generic"],
        },
    },

    # ---- CIMB -------------------------------------------------------------
    {
        "id": "S13_CIMB_BIZCHANNEL_IBG",
        "text": S13_CIMB_BIZCHANNEL_IBG,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "202608070343897254",
            "references": [
                ("202608070343897254", PRIMARY),
                ("OOL12612038565", PAYER),
            ],
            "bank_name": "CIMB",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "106.00",
            "fee": "0.10",
            "total_debit": "106.10",
        },
        "traps": {
            "reference_id": ["8000000001", "21000000000001"],
            # 106.10 is amount + fee; it is the total debit, not the amount.
            "amount": ["106.10", "0.10", "0.00", "8000000001"],
            "bank_name": ["RHB Bank", "RHB", "Unknown", "Generic"],
        },
        "notes": "THE fee/amount/total-debit discrimination case.",
    },
    {
        "id": "S14_CIMB_EMAIL_IBG",
        "text": S14_CIMB_EMAIL_IBG,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "202608050343665310",
            "references": [
                ("202608050343665310", PRIMARY),
                ("SNK22610700234", PAYER),
            ],
            "bank_name": "CIMB",
            "transaction_date": "2026-08-05",
            "amount": "31.80",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["1343"],
            "amount": ["202608050343665310"],
            "bank_name": ["Unknown", "Generic"],
        },
        "notes": "Email header date (August 5, 2026 7:02 PM) agrees with the "
                 "transaction date here, so it is not a trap on this sample.",
    },

    # ---- RHB --------------------------------------------------------------
    {
        "id": "S15_RHB_REFLEX",
        "text": S15_RHB_REFLEX,
        "ocr_used": False,
        "is_ibg": False,
        "expected": {
            "reference_id": "260806446749",
            "references": [
                ("260806446749", PRIMARY),
                ("NOVATECH BHD", PAYER),
                ("HAL22610717193", PAYER),
            ],
            "bank_name": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "74.20",
            "fee": "0.00",
            "total_debit": "74.20",
        },
        "traps": {
            "reference_id": ["20000000000001", "21000000000001", "100001A"],
            "amount": ["923103", "20000000000001", "0.00"],
            "bank_name": ["Unknown", "Generic"],
        },
        "notes": "Business Registration Number 100001A sits inside the money "
                 "block. Fund Transfer Charges and Service Tax Charges are "
                 "both fees; total_debit equals amount because both are zero.",
    },
    {
        "id": "S5_RHB_DEGRADED",
        "text": S5_RHB_DEGRADED,
        "ocr_used": True,
        "is_ibg": False,
        "expected": {
            "reference_id": None,
            "references": [],
            "bank_name": "RHB",
            "transaction_date": None,
            "amount": "10.69",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["RHBBMYKL"],
            "amount": ["12511799"],
        },
        "notes": "Degraded scan. The bar is: do not invent values.",
    },

    # ---- OCBC -------------------------------------------------------------
    {
        "id": "S16_OCBC_VELOCITY",
        "text": S16_OCBC_VELOCITY,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "20260806OCBCMYKL010OCB26583413",
            "references": [
                ("20260806OCBCMYKL010OCB26583413", PRIMARY),  # PayNet clearing ref
                ("SNK22610716537", PAYER),
                ("PGPY-69353 I2607427", PAYER),
                ("PGPY-69353", PAYER),
            ],
            "bank_name": "OCBC",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-06",
            "amount": "53.00",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["7300000001", "21000000000001"],
            "amount": ["7300000001", "21000000000001", "26583413"],
            "bank_name": ["RHB Bank", "RHB", "DuitNow", "Unknown", "Generic"],
        },
        "notes": "The only bank-issued reference is the PayNet one; there is "
                 "no separate OCBC transaction reference on this layout.",
    },
    {
        "id": "S17_OCBC_GIRO_REPORT",
        "text": S17_OCBC_GIRO_REPORT,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "MYGP260806823259",
            "references": [
                ("MYGP260806823259", PRIMARY),
                ("MYGP200414483476", PAYER),      # "Your Reference No"
                ("INV:CSE 22610716942", PAYER),
            ],
            "bank_name": "OCBC",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "752.60",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["7300000002", "21000000000001"],
            "amount": ["7300000002", "22610716942"],
            "bank_name": ["RHB Bank", "RHB", "Unknown", "Generic"],
        },
        "notes": "Amount prints as 'MYR - 752.60' with a stray hyphen; it is "
                 "not a negative number.",
    },

    # ---- Hong Leong -------------------------------------------------------
    {
        "id": "S18_HONGLEONG_DEBIT_ADVICE",
        "text": S18_HONGLEONG_DEBIT_ADVICE,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "C765070826120713",
            "references": [
                ("C765070826120713", PRIMARY),
                ("20260807HLBBMYKL010OCB65523848", SECONDARY),
                ("CMSPCT26080700011638", SECONDARY),
                ("DET-WHSU6510076", PAYER),
            ],
            "bank_name": "Hong Leong Bank",
            "transaction_date": "2026-08-07",
            "amount": "445.20",
            "fee": "0.00",
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["RHBBMYKL", "XXXXXXXXXX0001"],
            "amount": ["0.00", "65523848", "26080700011638"],
            "bank_name": ["RHB", "DuitNow", "Unknown", "Generic"],
        },
        "notes": "Four references, three roles. Two zero fee lines interleaved "
                 "with the reference lines.",
    },
    {
        "id": "S19_HONGLEONG_CREDIT_ADVICE",
        "text": S19_HONGLEONG_CREDIT_ADVICE,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "C773070826095112",
            "references": [
                ("C773070826095112", PRIMARY),
                ("IBGCMPRHBB2608070001189", SECONDARY),
                ("INV NO:12036941", PAYER),
                ("IA0060726", PAYER),
            ],
            # Credit advice: the ADVISING bank is Hong Leong. The payer banks
            # elsewhere; RHB is the beneficiary bank.
            "bank_name": "Hong Leong Bank",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "10.60",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["XXXXXXX0004", "XXXXXXXXXX0001"],
            "amount": ["12036941", "2608070001189"],
            "bank_name": ["RHB", "Unknown", "Generic"],
        },
        "notes": "Inbound credit advice -- issuer/beneficiary roles are "
                 "reversed relative to an outbound payment slip.",
    },

    # ---- UOB --------------------------------------------------------------
    {
        "id": "S20_UOB_INTERBANK_GIRO",
        "text": S20_UOB_INTERBANK_GIRO,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "FT26080178167822",
            "references": [
                ("FT26080178167822", PRIMARY),
                ("iv 10710913 10707642", PAYER),
                ("10707097 10706703", PAYER),
            ],
            "bank_name": "UOB",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "42.40",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["2000000001", "21000000000001", "100002186"],
            "amount": ["100002186", "2000000001", "10710913", "10707642"],
            "bank_name": ["RHB Bank", "RHB", "Unknown", "Generic"],
        },
        "notes": "Bank Code 100002186 is 9 digits -- passes the legacy "
                 "_is_plausible_amount gate.",
    },

    # ---- AmBank -----------------------------------------------------------
    {
        "id": "S21_AMBANK_DUITNOW",
        "text": S21_AMBANK_DUITNOW,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "FIN0708261152699",
            "references": [
                ("FIN0708261152699", PRIMARY),
                ("0708202611030843151", SECONDARY),   # Batch Ref Number
                ("DEPOSIT PAYMENT", PAYER),
            ],
            "bank_name": "AmBank",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "6010.60",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["8880000000001", "21000000000001"],
            "amount": ["8880000000001", "0708202611030843151"],
            "bank_name": ["RHB", "DuitNow", "Unknown", "Generic"],
        },
    },
    {
        "id": "S22_AMBANK_TRANSFER_ADVICE",
        "text": S22_AMBANK_TRANSFER_ADVICE,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "DTF20260803013679",
            "references": [
                ("DTF20260803013679", PRIMARY),
                ("MULTIPLE INVOICE", PAYER),
            ],
            "bank_name": "AmBank",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-03",
            "amount": "12587.50",
            "fee": "0.00",
            "total_debit": "12587.50",
        },
        "traps": {
            "reference_id": ["8880000000002", "21000000000001"],
            "amount": ["8880000000002", "0.00"],
            "bank_name": ["RHB", "DuitNow", "Unknown", "Generic"],
        },
        "notes": "Computer-generated stamp 03/08/2026 16:27:57 agrees with the "
                 "transaction date on this sample.",
    },

    # ---- Alliance ---------------------------------------------------------
    {
        "id": "S23_ALLIANCE_BIZSMART",
        "text": S23_ALLIANCE_BIZSMART,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "AOBIFT07082026075791",
            "references": [
                ("AOBIFT07082026075791", PRIMARY),
                ("OOL22610716770", PAYER),
                ("2330685513", PAYER),
            ],
            "bank_name": "Alliance Bank",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-07",
            "amount": "10.00",
            "fee": "0.00",
            "total_debit": "10.00",
        },
        "traps": {
            "reference_id": ["070000000000001", "21000000000001"],
            "amount": ["2330685513", "070000000000001", "0.00"],
            "bank_name": ["RHB", "DuitNow", "Unknown", "Generic"],
        },
        "notes": "Full label block then full value block, in matching order.",
    },

    # ---- HSBC -------------------------------------------------------------
    {
        "id": "S24_HSBC_PAYMENT_ADVICE",
        "text": S24_HSBC_PAYMENT_ADVICE,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "A2u38OFF1OT1",
            "references": [
                ("A2u38OFF1OT1", PRIMARY),
                ("9002490LZ2KO", SECONDARY),      # Instruction reference
                ("2536", PAYER),                  # Customer reference
                ("MED12612037415", PAYER),
            ],
            "bank_name": "HSBC",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-06",
            "amount": "31.80",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["212466600", "006718"],
            # "MYR -31.800" must not yield 31 or 31.800; and 04/08/2026 in the
            # invoice row is the document date, not the value date.
            "amount": ["31", "31.800", "-31.80", "006718", "2536"],
            "transaction_date": ["2026-08-07", "2026-08-04"],
            "bank_name": ["RHB", "Unknown", "Generic"],
        },
        "notes": "Negative, three-decimal invoice row. Advice sending date "
                 "(07 Aug) differs from value date (06 Aug).",
    },

    # ---- Standard Chartered ----------------------------------------------
    {
        "id": "S25_SCB_STRAIGHT2BANK",
        "text": S25_SCB_STRAIGHT2BANK,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "Q0483528",
            "references": [
                ("Q0483528", PRIMARY),                              # Bank Payment Reference
                ("e14ae93a-0499-4739-8ae9-3a0499173989", SECONDARY),  # UETR
                ("26080710nf880730729802000", SECONDARY),           # Import Reference
                ("C0166905", SECONDARY),                            # Batch Reference
                ("00117518", PAYER),                                # End to End ID
            ],
            "bank_name": "Standard Chartered",
            "beneficiary_bank": "RHB Islamic Bank",
            "transaction_date": "2026-08-07",
            "amount": "241.20",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["700000000001", "21000000000001", "SCBLMYKXXXX"],
            # Four invoice rows are money and must never win.
            "amount": ["120.00", "100.00", "10.60", "700000000001"],
            "transaction_date": ["2026-08-03", "2026-07-09", "2026-07-10"],
            "bank_name": ["RHB Islamic Bank", "RHB", "Unknown", "Generic"],
        },
        "notes": "Five bank references plus a four-row invoice table. "
                 "Beneficiary BIC is 11-char RHBAMYKLXXX (Islamic entity).",
    },

    # ---- Citibank ---------------------------------------------------------
    {
        "id": "S26_CITIBANK_CITIDIRECT",
        "text": S26_CITIBANK_CITIDIRECT,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "C50026080500004",
            "references": [
                ("C50026080500004", PRIMARY),
                ("db1e687ba63e453f947d2c2b72bf2936", SECONDARY),  # UETR
                ("4124014527", PAYER),
            ],
            "bank_name": "Citibank",
            "beneficiary_bank": "HSBC",
            # 08/05/2026 is 5 August: CitiDirect prints US month-first.
            "transaction_date": "2026-08-05",
            "amount": "120.00",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["110000001", "200000000001"],
            "amount": ["110000001", "200000000001", "4124014527", "297089"],
            # The day-first misreading. This is the whole point of the sample.
            "transaction_date": ["2026-05-08"],
            "bank_name": ["HSBC", "Unknown", "Generic"],
        },
        "notes": "THE month-first case. Every other sample is day-first.",
    },

    # ---- MUFG -------------------------------------------------------------
    {
        "id": "S27_MUFG_GCMS_IBG",
        "text": S27_MUFG_GCMS_IBG,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "G-GCP1803520",
            "references": [
                ("G-GCP1803520", PRIMARY),        # Group No.
                ("3644-GCP3705323", SECONDARY),   # Entry No.
                ("P79153", PAYER),
            ],
            "bank_name": "MUFG Bank (Malaysia)",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-03",
            "amount": "10982.12",
            "fee": None,
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["10000218", "21000000000001", "602000"],
            # Nine invoice rows are money.
            "amount": ["1817.90", "1,817.90", "82.08", "84.24", "1627.10",
                       "290.00", "10000218", "602000"],
            "transaction_date": ["2026-07-31", "2026-07-21", "2026-07-29"],
            "bank_name": ["RHB", "Unknown", "Generic"],
        },
        "notes": "Dotted-ISO value date (2026.08.03). Group total must beat "
                 "all nine invoice rows.",
    },

    # ---- Bank of China ----------------------------------------------------
    {
        "id": "S28_BANKOFCHINA_IGTB",
        "text": S28_BANKOFCHINA_IGTB,
        "ocr_used": False,
        "is_ibg": True,
        "expected": {
            "reference_id": "40002132227",
            "references": [
                ("40002132227", PRIMARY),
                ("EMC12612031446,2032331, YML12612033315", PAYER),
                ("YBP26080022", PAYER),
            ],
            "bank_name": "Bank of China (Malaysia)",
            "beneficiary_bank": "RHB",
            "transaction_date": "2026-08-06",
            "amount": "50.00",
            "fee": "0.00",
            "total_debit": None,
        },
        "traps": {
            "reference_id": ["100000000000001", "21000000000001"],
            "amount": ["0.00", "100000000000001", "40002132227"],
            "bank_name": ["RHB", "DuitNow", "Unknown", "Generic"],
        },
        "notes": "Three zero-value charge lines: Total Charges, Remittance "
                 "Commission, Tax Amount.",
    },
]


# --------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------

def by_id(sample_id: str) -> dict:
    """Fetch one corpus sample by its id."""
    for sample in CORPUS:
        if sample["id"] == sample_id:
            return sample
    raise KeyError("No corpus sample named %r" % (sample_id,))


def cases_for(field: str):
    """Yield (sample_id, text, ocr_used, expected_value, trap_values) per sample.

    `field` is one of: reference_id, bank_name, beneficiary_bank,
    transaction_date, amount, fee, total_debit.
    Use this to parametrize a field owner's test module.
    """
    for sample in CORPUS:
        yield (
            sample["id"],
            sample["text"],
            sample["ocr_used"],
            sample["expected"].get(field),
            sample.get("traps", {}).get(field, []),
        )


def reference_cases():
    """Yield (sample_id, text, ocr_used, expected_references) per sample.

    `expected_references` is a list of (value, role) tuples. Order is not
    significant; a test should compare as sets.
    """
    for sample in CORPUS:
        yield (
            sample["id"],
            sample["text"],
            sample["ocr_used"],
            sample["expected"].get("references", []),
        )


def samples_with(field: str):
    """Corpus samples where `field` has a non-None expectation."""
    return [s for s in CORPUS if s["expected"].get(field) is not None]
