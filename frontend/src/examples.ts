// One sample clause per label (same order as LABELS in src/config.py).
// Written for the demo, not copied from LEDGAR, so they are not test data.

export type Example = {
  label: string
  text: string
}

export const EXAMPLES: Example[] = [
  {
    label: 'Governing Law',
    text: 'This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without giving effect to any choice of law or conflict of law rules that would cause the application of the laws of any other jurisdiction.',
  },
  {
    label: 'Termination',
    text: 'Either party may terminate this Agreement upon thirty (30) days prior written notice if the other party materially breaches any provision of this Agreement and fails to cure such breach within such thirty-day period.',
  },
  {
    label: 'Confidentiality',
    text: 'The Receiving Party shall hold all Confidential Information of the Disclosing Party in strict confidence, shall not disclose it to any third party, and shall use it solely for the purpose of performing its obligations under this Agreement.',
  },
  {
    label: 'Indemnification',
    text: 'The Supplier shall indemnify, defend and hold harmless the Customer and its officers, directors and employees from and against any and all losses, damages, liabilities and expenses, including reasonable attorneys fees, arising out of any breach of this Agreement by the Supplier.',
  },
  {
    label: 'Notices',
    text: 'All notices, requests and other communications under this Agreement shall be in writing and shall be deemed duly given when delivered by hand, by overnight courier, or by email with confirmation of transmission, to the addresses set forth on the signature page.',
  },
  {
    label: 'Severability',
    text: 'If any term or provision of this Agreement is held to be invalid, illegal or unenforceable in any jurisdiction, such invalidity shall not affect any other term or provision of this Agreement, which shall remain in full force and effect.',
  },
  {
    label: 'Assignment',
    text: 'Neither party may assign or transfer this Agreement or any of its rights or obligations hereunder without the prior written consent of the other party, except to a successor in connection with a merger or a sale of all or substantially all of its assets.',
  },
  {
    label: 'Entire Agreement',
    text: 'This Agreement, together with its exhibits, constitutes the entire agreement between the parties with respect to its subject matter and supersedes all prior and contemporaneous agreements, proposals and understandings, whether written or oral.',
  },
]  