from tkinter import *

wn = Tk()
wn.title("Test Window")
wn.geometry("300x200")
Label(wn, text="Tkinter is working!").pack(pady=20)
wn.mainloop()

 #Function to get the meaning of a word using NLTK's WordNet
def meaning():
    # Taking the string input
    query = str(text.get())
    synsets = wordnet.synsets(query)
    res = ''
    
    if synsets:
        # Fetch the definition of the first meaning
        for syn in synsets:
            res += f"{syn.definition()}\n"
        
        # Set and speak the output
        spokenText.set(res)
        speak("The meaning is: " + res)
    else:
        res = "Meaning not found"
        spokenText.set(res)
        speak(res)

# Creating the window 
wn = tk() 
wn.title("Senapati's Dictionary")
wn.geometry('700x500')
wn.config(bg='SlateGray1')