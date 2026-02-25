# Rag-Assesment
assesment 2 AI engineer role

POST - /upload
    body -> give pdf file
    return -> document_id, page_count

POST - /query
    body -> quection
    return -> acknowlegment


How to Setup :
    create conda environment and setup the environment.
    
    then run separate fastapi backends for backend, call_back_backend. use separate ports and remeber
        backend 
            Here do the logics
        call_back_backend
            used to capture the call_back url
    
    open swagger UI.

    using /upload API call send the given pdf. (2 - 3 minutes to create vector db)

    it will returns the document_id saves it.

    use /query API call replace these values in that body
            "document_id": replace documentation id,
            "question": ask quection,
            "callback_url": replace call_back_receiver url + callback,  eg:- http:localhost:8001/callback
            "top_k": replace number top similars
    
    this will return immediate acknowlegment and you can see the response in console in call_back_backend console