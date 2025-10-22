(about-concept-a)=
# [Concept/Topic Name] Concepts

[Opening paragraph that introduces the topic and provides context. Explain what will be covered on this page.]

## [Main Concept 1]

[Detailed explanation of the first major concept. Use paragraphs to explain complex ideas clearly.]

[Additional paragraph with more details, examples, or context.]

### [Sub-concept or Detail]

[More detailed explanation of a specific aspect of the main concept.]

- **[Term or aspect]**: [Explanation]
- **[Term or aspect]**: [Explanation]

### [Visual Representation]

[Optional: Introduction to a diagram or visual aid]

```{mermaid}
---
caption: [Diagram Title]
---
flowchart LR
    A[Start] --> B{Decision}
    B -->|Option 1| C[Result 1]
    B -->|Option 2| D[Result 2]
    C & D --> E[End]

    classDef defaultClass fill:#fff,stroke:#000,stroke-width:1px
    class A,B,C,D,E defaultClass
```

[Explanation of what the diagram shows and its key takeaways.]

(reference-anchor-name)=
## [Main Concept 2]

[Detailed explanation with technical depth. Use paragraphs for explanation.]

### [Implementation Details]

[Description of how this works in practice.]

![Diagram description](./_images/diagram-filename.png)

[Explanation of the diagram and its significance.]

### [Process Steps]

During [process name]:

1. [Step 1 description]
2. [Step 2 description]
3. [Step 3 description]
4. [Step 4 description]

This approach:

- [Benefit 1]
- [Benefit 2]
- [Benefit 3]
- [Benefit 4]

Additional Resources:

- [External resource title](https://external-url.com)
- [Paper or documentation title](https://external-url.com)
- [Tutorial or example](https://external-url.com)

## [Main Concept 3]

[Introduction to this concept with context.]

### [Sub-topic]

[Detailed explanation with technical specifics.]

:::{note}
[Important note or callout for users to be aware of. Use admonitions to highlight crucial information.]
:::

### [Data Format or Structure Example]

[Introduction to the format or structure being explained.]

#### [Use Case Example 1]

[Description of the use case and what it demonstrates.]

Format:

```text
prompt: "[Example prompt template with <placeholders>]"
completion: "[Example completion]"
```

#### [Use Case Example 2]

[Description of another use case.]

Format:

```text
prompt: "[Example prompt template]"
completion: "[Example completion]"
```

### [Alternative Format or Approach]

[Description of an alternative approach or format.]

For more information refer to [linked documentation](../../path/to/other-doc.md#section-anchor).

## [Configuration or Parameters]

[Introduction to configuration options or parameters.]

```{list-table}
:header-rows: 1

* - Parameter
  - Description
  - Recommended Value
* - [Parameter Name]
  - [Detailed description of what this parameter controls and its impact.]
  - [Recommended value or range with context]
* - [Parameter Name]
  - [Detailed description with technical details.]
  - [Recommended value with range like `1e-3` to `1e-5`]
* - [Parameter Name]
  - [Description including when to use different values.]
  - [Recommended starting point and guidance]
* - [Parameter Name]
  - [Description with formula or calculation if relevant.]
  - [Recommendation with technical context]
```

## [Technical Implementation Details]

[Introduction to technical implementation aspects.]

### [Implementation Approach 1]

[Description of the approach and its purpose. Include links to external documentation using standard format.]

[Detailed technical explanation with specifics.]

Can be configured via `parameter_name` in the [Configuration Documentation](../../../docs/path/to/config.md).

:::{note}
[Technical note about limitations, compatibility, or version-specific behavior.]
:::

### [Implementation Approach 2]

[Description and explanation.]

Can be configured via `parameter_name` in the [Configuration Documentation](../../../docs/path/to/config.md).

#### Configuration

- Constraints
  - [Constraint description with technical details]
  - [Additional constraints or requirements]
  
- Multi-component considerations
  - [Technical consideration with detailed explanation]
  - [Additional considerations with examples]
  
  Example: [Detailed example scenario with specific numbers and rationale]

- Performance
  - [Performance consideration #1]
  - [Performance consideration #2]

## [Advanced Feature]

[Description of an advanced or optional feature.]

[Explanation of benefits]:

- [Benefit #1]
- [Benefit #2]
- [Benefit #3]

[Additional context about when to use this feature.]

### Limitations

- [Limitation description with specifics]:
  - [Specific case or model]
  - [Specific case or model]
  - [Specific case or model]

- [Another limitation with details]

:::{note}
[Warning or important note about limitations or behavior.]
:::

### Example of using in the API

[Introduction to the code example.]

```bash
curl --location \
"https://$HOSTNAME/v1/api/endpoint" \
--header 'Accept: application/json' \
--header 'Content-Type: application/json' \
--data '{
    "param1": "value1",
    "param2": {
        "nested_param": true,
        "another_param": 10,
        "numeric_param": 0.00001
    }
}' | jq
```

Learn how to [accomplish related task] by following the [Tutorial Name](../../path/to/tutorial.md) tutorial.

---

## Comparison Table Format

Use list-tables for detailed comparisons:

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Use Case
  - Approach A
  - Approach B
* - **[Aspect 1]**
  - [Description of how Approach A handles this]
  - [Description of how Approach B handles this]
* - **[Aspect 2]**
  - [Description with technical details]
  - [Description with technical details]
* - **[Aspect 3]**
  - [Description]
  - [Description]
```

