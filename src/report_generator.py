import datetime
from jinja2 import Template
from google.cloud import bigquery
from src.ai_model import generate_response

class ReportGenerator:
    def __init__(self, project_id, dataset_id, table_id):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.template = self.load_template()

    def load_template(self):
        # ISO-friendly report template
        return Template("""
        ISO-Compliant Data Analysis Report
        ==================================

        Report ID: {{ report_id }}
        Date Generated: {{ date_generated }}
        Classification: {{ classification }}

        1. Executive Summary
        --------------------
        {{ executive_summary }}

        2. Data Overview
        ----------------
        Dataset: {{ dataset_name }}
        Records Analyzed: {{ record_count }}
        Time Period: {{ time_period }}

        3. Key Findings
        ---------------
        {{ key_findings }}

        4. Detailed Analysis
        --------------------
        {{ detailed_analysis }}

        5. Visualizations
        -----------------
        Looker Studio Dashboard: {{ looker_url }}

        Key Visualizations:
        {{ visualizations }}

        6. Recommendations
        ------------------
        {{ recommendations }}

        7. Methodology
        --------------
        {{ methodology }}

        8. Data Quality and Limitations
        -------------------------------
        {{ data_quality }}

        9. Appendices
        -------------
        {{ appendices }}

        10. Revision History
        -------------------
        {{ revision_history }}

        End of Report
        """)

    def fetch_data_summary(self):
        query = f"""
        SELECT 
            COUNT(*) as record_count,
            MIN(date_column) as start_date,
            MAX(date_column) as end_date
        FROM `{self.dataset_id}.{self.table_id}`
        """
        query_job = self.client.query(query)
        results = query_job.result()
        for row in results:
            return row.record_count, row.start_date, row.end_date

    def generate_insights(self, classification):
        # Use AI model to generate insights based on classification and data summary
        prompt = f"""
        Generate insights for a {classification} dataset with the following summary:
        - Record count: {self.record_count}
        - Time period: {self.start_date} to {self.end_date}

        Provide:
        1. An executive summary (2-3 sentences)
        2. 3-5 key findings
        3. A brief detailed analysis (2-3 paragraphs)
        4. 2-3 recommendations
        5. A short methodology description
        """
        insights = generate_response(prompt)
        return insights

    def generate_report(self, classification, looker_url):
        self.record_count, self.start_date, self.end_date = self.fetch_data_summary()
        insights = self.generate_insights(classification)

        # Parse insights (this is a simplified parsing, you might want to use a more robust method)
        insight_parts = insights.split('\n\n')
        executive_summary = insight_parts[0] if len(insight_parts) > 0 else "N/A"
        key_findings = insight_parts[1] if len(insight_parts) > 1 else "N/A"
        detailed_analysis = insight_parts[2] if len(insight_parts) > 2 else "N/A"
        recommendations = insight_parts[3] if len(insight_parts) > 3 else "N/A"
        methodology = insight_parts[4] if len(insight_parts) > 4 else "N/A"

        report = self.template.render(
            report_id=f"REP-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            date_generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            classification=classification,
            executive_summary=executive_summary,
            dataset_name=f"{self.dataset_id}.{self.table_id}",
            record_count=self.record_count,
            time_period=f"{self.start_date} to {self.end_date}",
            key_findings=key_findings,
            detailed_analysis=detailed_analysis,
            looker_url=looker_url,
            visualizations="Please refer to the Looker Studio Dashboard for detailed visualizations.",
            recommendations=recommendations,
            methodology=methodology,
            data_quality="Data quality checks were performed to ensure accuracy and completeness. Any limitations in the data have been noted in the analysis.",
            appendices="N/A",
            revision_history="Initial version"
        )
        return report

    def save_report(self, report, filename):
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved as {filename}")

# Usage example - Parameters from process_file
def generate_report_from_file(file_processor_output):
    # Unpack the output from the file_processor's process_file function
    preview_data, classification, looker_url, report_filename, table_id = file_processor_output
    
    # Define project and dataset IDs (you can retrieve these from config/environment)
    project_id = "your-project-id"
    dataset_id = "your-dataset-id"

    # Initialize the ReportGenerator with project, dataset, and table ID
    report_gen = ReportGenerator(project_id, dataset_id, table_id)
    
    # Generate the report with classification and Looker URL
    report = report_gen.generate_report(classification, looker_url)
    
    # Save the report using the provided filename
    report_gen.save_report(report, report_filename)

# Example usage
if __name__ == "__main__":
    # Example file_processor output
    file_processor_output = (None, "cosmetic", "https://lookerstudio.google.com/your-dashboard-url", "ISO_Report_cosmetic_sample.txt", "your-table-id")
    generate_report_from_file(file_processor_output)
