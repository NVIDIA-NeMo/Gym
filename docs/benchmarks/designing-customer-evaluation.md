(designing-customer-evaluation)=

# Designing a customer evaluation benchmark

## Guiding Principles

Designing a customer evaluation benchmark is most similar to product design rather than strictly engineering or research. The majority of the work involves building an understanding of what the customer’s system is actually intended to do in production and how they intend their users to use it, and then capturing that reality as faithfully as possible. Specifically, we want to understand what the customer’s users actually care about. What are they trying to accomplish? What counts as success? What kinds of failures matter, and which ones are tolerable?

The only goal of a customer eval is to reflect the real downstream use case so that we can get a high-fidelity signal of model performance on how people are actually using the model.

Customer eval design is not a research problem. For example, you should not be doing things like cleaning up the data, removing noisy inputs, generating synthetic examples, and building something that looks nice and tidy. While that may be close to the customer setting, it usually is not and our goal is to match the downstream use case distribution as closely as possible. If the customer uses messy scraped documents in production, your eval should use messy scraped documents. If their prompts are awkward or inconsistent, your eval should reflect that too. In some sense, we are not trying to build the “ideal” system; we are just trying to mirror reality, which is significantly harder.

The dream scenario for a customer eval setup is that the customer gives you a year of production traffic, along with the exact model inputs, labels, and their scoring logic. At that point, you are basically doing data science. In reality, that almost never happens. More often, customers will not have a real eval set or the eval is synthetic without principled design decisions, the distribution is skewed and not reflective of downstream use case distribution, labels are incomplete or inaccurate, and the evaluation and scoring logic is not well defined.

If the customer hands you an eval set, do not assume it is high quality. You need to understand how it was created and whether it actually reflects production behavior. For example, if 10 samples all share the same failure pattern, is that really how the system fails in production? Or did someone just over-sample a known issue?

## Functional Tips

Whenever you work with a customer, try to get three things upfront:

1. Raw model inputs exactly as they are sent to the endpoint, that is, the actual chat completion payloads. That gives you visibility into prompt structure, context formatting, and any preprocessing happening behind the scenes. If they cannot give you that format, convert what they provide and ask them to verify it.  
2. Scoring logic or evaluation harness. How do they decide if an output is good or bad? What metrics matter? If they won’t share labels, you may need to create your own and then ask them to audit and approve them.  
3. Spend real time understanding the downstream use case. Even if they give you data, you still need context. What are their users expecting? What is business-critical? What failure actually hurts them?
