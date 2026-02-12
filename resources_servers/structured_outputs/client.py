# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="structured_outputs",
    url_path="/verify",
    json={
        "responses_create_params": {
            "input": [
                {
                    "content": '# Information\nThe family volunteer activity titled "Park Cleanup Day" is hosted by the Green Earth Organization and will take place on September 16, 2023, at Riverside Park in Springfield, Illinois. The park is located at geographic coordinates latitude 39.7817 and longitude -89.6501. The event is scheduled to last for 3 hours and is designed to be child-friendly, though it is weather-dependent and may be canceled in the event of heavy rain. The Green Earth Organization contact for the event is Jane Miller, who can be reached at +15551234567 or via email at jane.miller@greenearth.org. The organization\u2019s physical address is 123 Eco Street, Springfield, IL, 62701, United States.\n\nParticipating family members from the Lopez household include:\n- Maria Lopez, age 38, serving in a parent role and confirmed as available. She brings valuable skills in gardening and first aid.\n- James Lopez, age 14, participating as a teenager and also available. He has prior experience with basic cleanup tasks.\n- Elena Lopez, age 68, joining as a grandparent and available. She has a background in community organizing, which may assist in coordinating group efforts during the event.\n\nPreparation for the activity includes several assigned tasks and logistical considerations. Maria Lopez is responsible for confirming the family\u2019s attendance by September 10, 2023. James is tasked with packing the necessary supplies by September 15, 2023. Maria has already completed the review of safety guidelines, ensuring the family is aware of protocols during the cleanup. \n\nSupplies required for the event are a mix of organization-provided and family-provided items:\n- The Green Earth Organization will supply 5 trash grabbers.\n- The Lopez family will bring 10 garbage bags.\n- First aid kits will be shared between the family and the organization.\n\nNo special health concerns have been reported: there are no known allergies among the participants, and no mobility issues have been identified. The event is structured to accommodate participants of various ages and experience levels, promoting inclusive and safe community engagement. Given the outdoor nature of the activity, participants are advised to monitor weather forecasts leading up to the event, as heavy rain could lead to cancellation.\n\n# Problem: Extract the information from the text and format it as a JSON object matching this schema.',
                    "role": "user",
                    "type": "message",
                },
                {
                    "content": 'Create a structured JSON response that:\n1. Implements proper data typing\n2. Handles special characters appropriately\n3. Includes all required fields\n4. Maintains proper nesting levels\n5. Provides appropriate array formatting\n6. Validates against schema constraints\n7. Uses consistent formatting and escaping\n8. Is unindented JSON format\n9. Is grounded in the provided dialog\n10. Strictly follows the provided schema: {"activityName": {"type": "string", "minLength": 1, "description": "Name of the volunteer activity."}, "organization": {"type": "object", "required": ["name", "contact", "address"], "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "contact": {"type": "object", "required": ["person", "phone", "email"], "additionalProperties": false, "properties": {"person": {"type": "string", "minLength": 1}, "phone": {"type": "string", "pattern": "^\\\\+?[1-9]\\\\d{1,14}$"}, "email": {"type": "string", "format": "email"}}}, "address": {"type": "object", "required": ["street", "city", "state", "zipCode", "country"], "additionalProperties": false, "properties": {"street": {"type": "string", "minLength": 1}, "city": {"type": "string", "minLength": 1}, "state": {"type": "string", "minLength": 1}, "zipCode": {"type": "string", "pattern": "^\\\\d{5}(-\\\\d{4})?$|^\\\\w\\\\d\\\\w ?\\\\d\\\\w\\\\d$"}, "country": {"type": "string", "minLength": 1}}}}}, "date": {"type": "string", "format": "date", "description": "Planned date of the volunteer activity."}, "location": {"type": "object", "required": ["venue", "coordinates"], "additionalProperties": false, "properties": {"venue": {"type": "string", "minLength": 1}, "coordinates": {"type": "object", "required": ["latitude", "longitude"], "additionalProperties": false, "properties": {"latitude": {"type": "number", "minimum": -90, "maximum": 90}, "longitude": {"type": "number", "minimum": -180, "maximum": 180}}}}}, "familyParticipants": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["name", "age", "role", "availability"], "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0}, "role": {"type": "string", "enum": ["parent", "child", "grandparent", "guardian", "teenager"]}, "availability": {"type": "boolean"}, "skills": {"type": "array", "items": {"type": "string"}, "uniqueItems": true}}}}, "activityType": {"type": "string", "enum": ["communityCleanup", "foodDistribution", "elderlyCompanionship", "tutoring", "animalCare", "eventSupport"], "description": "Category of volunteer work."}, "durationHours": {"type": "number", "minimum": 0.5, "maximum": 12, "description": "Planned duration in hours."}, "preparationTasks": {"type": "array", "items": {"type": "object", "required": ["task", "assignedTo", "dueDate", "completed"], "additionalProperties": false, "properties": {"task": {"type": "string", "minLength": 1}, "assignedTo": {"type": "string", "minLength": 1}, "dueDate": {"type": "string", "format": "date"}, "completed": {"type": "boolean"}}}}, "suppliesNeeded": {"type": "array", "items": {"type": "object", "required": ["item", "quantity", "providedBy"], "additionalProperties": false, "properties": {"item": {"type": "string", "minLength": 1}, "quantity": {"type": "integer", "minimum": 1}, "providedBy": {"type": "string", "enum": ["family", "organization", "shared", "purchased"]}}}}, "specialConsiderations": {"type": "object", "required": ["allergies", "mobilityIssues", "childFriendly", "weatherDependent"], "additionalProperties": false, "properties": {"allergies": {"type": "array", "items": {"type": "string"}}, "mobilityIssues": {"type": "boolean"}, "childFriendly": {"type": "boolean"}, "weatherDependent": {"type": "boolean"}, "notes": {"type": "string"}}}, "completed": {"type": "boolean", "description": "Indicates if the activity has been completed."}, "feedback": {"type": "object", "required": ["familyRating", "organizationFeedback", "lessonsLearned"], "additionalProperties": false, "properties": {"familyRating": {"type": "integer", "minimum": 1, "maximum": 5}, "organizationFeedback": {"type": "string"}, "lessonsLearned": {"type": "array", "items": {"type": "string"}}, "photosTaken": {"type": "boolean"}}}}',
                    "role": "user",
                    "type": "message",
                },
            ]
        },
        "response": {
            "id": "response_1",
            "created_at": 1.0,
            "model": "model_1",
            "object": "response",
            "output": [
                {
                    "id": "message_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": '{"activityName":"Park Cleanup Day","organization":{"name":"Green Earth Organization","contact":{"person":"Jane Miller","phone":"+15551234567","email":"jane.miller@greenearth.org"},"address":{"street":"123 Eco Street","city":"Springfield","state":"IL","zipCode":"62701","country":"United States"}},"date":"2023-09-16","location":{"venue":"Riverside Park","coordinates":{"latitude":39.7817,"longitude":-89.6501}},"familyParticipants":[{"name":"Maria Lopez","age":38,"role":"parent","availability":true,"skills":["gardening","first aid"]},{"name":"James Lopez","age":14,"role":"teenager","availability":true,"skills":["basic cleanup"]},{"name":"Elena Lopez","age":68,"role":"grandparent","availability":true,"skills":["community organizing"]}],"activityType":"communityCleanup","durationHours":3,"preparationTasks":[{"task":"Confirm family attendance","assignedTo":"Maria Lopez","dueDate":"2023-09-10","completed":false},{"task":"Pack necessary supplies","assignedTo":"James Lopez","dueDate":"2023-09-15","completed":false},{"task":"Review safety guidelines","assignedTo":"Maria Lopez","dueDate":"2023-09-10","completed":true}],"suppliesNeeded":[{"item":"trash grabber","quantity":5,"providedBy":"organization"},{"item":"garbage bag","quantity":10,"providedBy":"family"},{"item":"first aid kit","quantity":2,"providedBy":"shared"}],"specialConsiderations":{"allergies":[],"mobilityIssues":false,"childFriendly":true,"weatherDependent":true,"notes":"Monitor weather forecasts. Event may be canceled due to heavy rain. No reported allergies or mobility issues."},"completed":false,"feedback":{"familyRating":1,"organizationFeedback":"","lessonsLearned":[],"photosTaken":false}}',
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        "schema_str": '{"type": "object", "required": ["projectId", "projectName", "category", "materials", "steps", "estimatedDuration", "difficultyLevel", "isCompleted", "creator", "lastModified"], "properties": {"projectId": {"type": "string", "description": "Unique identifier for the craft project"}, "projectName": {"type": "string", "description": "Name of the craft project"}, "category": {"type": "string", "enum": ["knitting", "painting", "scrapbooking", "woodworking", "origami", "jewelry"], "description": "Main craft category"}, "materials": {"type": "array", "items": {"type": "object", "required": ["name", "quantity", "unit", "isPurchased"], "properties": {"name": {"type": "string"}, "quantity": {"type": "number", "minimum": 0}, "unit": {"type": "string", "enum": ["piece", "meter", "gram", "ounce", "liter", "unit"]}, "isPurchased": {"type": "boolean", "description": "Whether the material has been purchased"}, "supplier": {"type": "object", "required": ["name", "contact"], "properties": {"name": {"type": "string"}, "contact": {"type": "string", "format": "email"}, "website": {"type": "string", "format": "uri"}}, "additionalProperties": false}}, "additionalProperties": false}}, "steps": {"type": "array", "items": {"type": "object", "required": ["stepNumber", "instruction", "estimatedTimeMinutes"], "properties": {"stepNumber": {"type": "integer", "minimum": 1}, "instruction": {"type": "string", "description": "Detailed step instruction"}, "estimatedTimeMinutes": {"type": "integer", "minimum": 1}, "toolsRequired": {"type": "array", "items": {"type": "object", "required": ["toolName", "quantity"], "properties": {"toolName": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1}, "isAvailable": {"type": "boolean"}}, "additionalProperties": false}}, "visualAid": {"type": "object", "required": ["url", "caption"], "properties": {"url": {"type": "string", "format": "uri"}, "caption": {"type": "string"}, "thumbnail": {"type": "object", "required": ["url", "width", "height"], "properties": {"url": {"type": "string", "format": "uri"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "additionalProperties": false}}, "additionalProperties": false}}, "additionalProperties": false}}, "estimatedDuration": {"type": "integer", "description": "Total estimated time to complete the project"}, "difficultyLevel": {"type": "string", "enum": ["beginner", "intermediate", "advanced"], "description": "Skill level required"}, "isCompleted": {"type": "boolean", "description": "Whether the project has been finished"}, "creator": {"type": "object", "required": ["name", "contactEmail"], "properties": {"name": {"type": "string"}, "contactEmail": {"type": "string", "format": "email"}, "profile": {"type": "object", "required": ["skillLevel", "joinedDate"], "properties": {"skillLevel": {"type": "string", "enum": ["novice", "enthusiast", "expert"]}, "joinedDate": {"type": "string", "format": "date"}, "badges": {"type": "array", "items": {"type": "string"}}}, "additionalProperties": false}}, "additionalProperties": false}, "lastModified": {"type": "string", "format": "date-time", "description": "Timestamp of last update"}}, "additionalProperties": false}',
        "schema_type": "json",
    },
)
result = asyncio.run(task)
print(json.dumps(asyncio.run(result.json()), indent=4))
