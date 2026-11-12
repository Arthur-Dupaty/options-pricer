from datetime import date
import yfinance as yf

t = yf.Ticker("^SPX")
echeances = t.options
print("Nombre d'échéances :", len(echeances))
print("Premières échéances :", echeances[:10])

# première échéance à au moins 3 semaines
echeance = next(e for e in echeances if (date.fromisoformat(e) - date.today()).days >= 21)
chaine = t.option_chain(echeance)
calls, puts = chaine.calls, chaine.puts

cols = ["strike", "bid", "ask", "impliedVolatility", "openInterest"]
print("\nÉchéance choisie :", echeance)
print("Calls :", len(calls), "lignes | Puts :", len(puts), "lignes")
print(calls[cols].head(15))
print(puts[cols].head(15))