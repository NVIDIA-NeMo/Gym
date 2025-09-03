import os
import pandas as pd


class CompanyDirectoryTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            current_dir, "..", "csv_data", "raw", "email_addresses.csv"
        )
        self._emails = pd.read_csv(data_path, header=None, names=["email_address"])

    def find_email_address(self, name=""):
        """
        Find email addresses containing the given name (case-insensitive).

        Args:
            name: Name or partial name to search for

        Returns:
            List of matching email addresses or a message if name not provided
        """
        if name == "":
            return "Name not provided."
        name = name.lower()
        email_address = self._emails[
            self._emails["email_address"].str.contains(name, case=False)
        ]
        return email_address["email_address"].values.tolist()


schema_find_email_address = {
    "type": "function",
    "name": "company_directory_find_email_address",
    "description": "Finds all email addresses containing the given name (case-insensitive search).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name or partial name to search for in email addresses",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

# List of all schemas defined in this file
company_directory_tool_schemas = [schema_find_email_address]
