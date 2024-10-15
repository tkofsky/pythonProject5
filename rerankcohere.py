# Make sure to have the newest Cohere SDK installed:
# pip install -U cohere
# Get your free API key from: www.cohere.com

import cohere

cohere_key = "1ODOOnJTr2xY7ncwJP0f54wZaA8o5AmB9hJQmHq7"
co = cohere.Client(cohere_key)

# Lets define some JSON with our documents. Here we use a JSON to represent emails
# with different fields. In the call to co.rerank we can specify which
emails = [
    {
        "from": "Paul Doe <paul_fake_doe@oracle.com>",
        "to": ["Steve <steve@me.com>", "lisa@example.com"],
        "date": "2024-03-27",
        "subject": "Follow-up",
        "text": "We are happy to give you the following pricing for your project."
    },
    {
        "from": "John McGill <john_fake_mcgill@microsoft.com>",
        "to": ["Steve <steve@me.com>"],
        "date": "2024-03-28",
        "subject": "Missing Information",
        "text": "Sorry, but here is the pricing you asked for for the newest line of your models."
    },
    {
        "from": "John McGill <john_fake_mcgill@microsoft.com>",
        "to": ["Steve <steve@me.com>"],
        "date": "2024-02-15",
        "subject": "Commited Pricing Strategy",
        "text": "I know we went back and forth on this during the call but the pricing for now should follow the agreement at hand."
    },
    {
        "from": "Generic Airline Company<no_reply@generic_airline_email.com>",
        "to": ["Steve <steve@me.com>"],
        "date": "2023-07-25",
        "subject": "Your latest flight travel plans",
        "text": "Thank you for choose to fly Generic Airline Company. Your booking status is confirmed."
    },
    {
        "from": "Generic SaaS Company<marketing@generic_saas_email.com>",
        "to": ["Steve <steve@me.com>"],
        "date": "2024-01-26",
        "subject": "How to build generative AI applications using Generic Company Name",
        "text": "Hey Steve! Generative AI is growing so quickly and we know you want to build fast!"
    },
    {
        "from": "Paul Doe <paul_fake_doe@oracle.com>",
        "to": ["Steve <steve@me.com>", "lisa@example.com"],
        "date": "2024-04-09",
        "subject": "Price Adjustment",
        "text": "Re: our previous correspondence on 3/27 we'd like to make an amendment on our pricing proposal. We'll have to decrease the expected base price by 5%."
    },
]

# Define which fields we want to include for the ranking:
rank_fields = ["from", "to", "date", "subject", "body"]

# To get all fields, you can also call: rank_fields = list(docs[0].keys())

# Define a query. Here we ask for the pricing from Mircosoft (MS).
# The model needs to combine information from the email (john_fake_mcgill@microsoft.com>)
# and the body


# Call rerank, pass in the query, docs, and the rank_fields. Set the model to 'rerank-english-v3.0' or 'rerank-multilingual-v3.0'
results = co.rerank(query=query, documents=emails, top_n=2, model='rerank-english-v3.0')

print("Query:", query)
for hit in results.results:
    email = emails[hit.index]
    print(email)

# Now we ask for the pricing from Oracle
query = "Which pricing did we get from MS?"

# Call rerank, pass in the query, docs, and the rank_fields. Set the model to 'rerank-english-v3.0' or 'rerank-multilingual-v3.0'
results = co.rerank(query=query, documents=emails, top_n=2, model='rerank-english-v3.0', rank_fields=rank_fields)

print("Query:", query)
for hit in results.results:
    email = emails[hit.index]
    print(email)