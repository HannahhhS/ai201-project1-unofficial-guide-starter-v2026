import questions

def judge(question: str, expects: str, answer: str, results) -> bool:
    if not expects:
        return False
    return expects.strip().lower() in (answer or "".lower) #returns true when asnwer contaisna phrase the correct asnwer has to contain