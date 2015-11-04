#!/usr/bin/python

#############################
# Wordnet Labeler GUI      #
# Created by Yuan Wang     #
# Originally for affordnace #
# Modified on 11/03/2015   #
#############################



from Tkinter import *
import nltk
import json
import pickle


###########
# change path here to load dictionaries to label


### load inherited affordance dictionary
affordance_dict_fpath = '/Users/AriaW/Desktop/159/project-beta/code/utils/description_pp/sample_description.json'

"""
Load json file should be an dictionry with the keys as the words to be tagged, 
and the values to be WordNet tags;
Original created by Yuan Wang to tagged affordance using WordNet; 


"""



# store_fpath = '/Users/AriaW/Desktop/GallantLab/Affordance/store.py'
# store_fpath = '/Users/AriaW/Desktop/GallantLab/Affordance/other_codes/stored.py'

with open(affordance_dict_fpath) as fid:
    affordanceDict = json.loads(fid.readlines()[0])
fid.close()
verbMeanings = ['input needed']
meaningList = []
# storeList = [str(k) + ": " + str(v) for k,v in storeDict.items()]

def get_wn_synsets(lemma):
   """Get all synsets for a word, return a list of [wordnet_label,definition, hypernym_string]
   for all synsets returned."""
   from nltk.corpus import wordnet as wn
   synsets = wn.synsets(lemma)
   out = []
   for s in synsets:
       # if not '.v.' in s.name(): continue # only verbs!
       hyp = ''
       for ii,ss in enumerate(s.hypernym_paths()):
           try:
               hyp+=(repr([hn.name() for hn in ss])+'\n')
           except:
               hyp+='FAILED for %dth hypernym\n'%ii
       out.append(dict(synset=s.name(), definition=s.definition(),hypernyms=hyp))
   return out

def get_wn_meaning(lemma):
    """get meaning of a word using wordNet labels"""
    # from nltk.corpus import wordnet as wn
    # return wn.synset(lemma).definition()
    return None
                          
def print_hyp(lemma):
	global verbMeanings, meaningList
	verbMeanings = []
	meaningList = get_wn_synsets(lemma)
	for x in meaningList:
	    verbMeanings += [x['synset'] + ':' + "     " + x['definition']]
	return verbMeanings

def nounSelected () :
    print "At %s of %d" % (select.curselection(), len(affordanceDict.keys()))
    return int(select.curselection()[0])

def meaningSelected () :
    print "At %s of %d" % (select6.curselection(), len(verbMeanings))
    return int(select6.curselection()[0])

# def storeSelected () :
#     print "At %s of %d" % (select8.curselection(), len(storeList))
#     return int(select8.curselection()[0])

def count():
    count = 0
    for k in affordanceDict.keys():
        if affordanceDict[k] == []:
            count+=1
    return str(count) + "/" + str(len(affordanceDict.keys()))

def addEntry (input=None) :
    if nounVar.get() != None:
        affordanceDict[nounVar.get()] = affVar.get().split(",")
        setSelect ()
    msg = nounVar.get() +  " ADDED  " + count() + " to go" 
    msgVar.set(msg)

def lookUp (input=None):
    verb = verbVar.get()
    global verbmeanings
    verbmeanings = print_hyp(verb)
    setSelect6 ()
    
def loadEntry (input=None) :
    noun = affordanceDict.keys()[nounSelected()]
    nounVar.set(noun)
    verbVar.set(noun)
    defVar.set(get_wn_meaning(noun))
    try:
        aff = ','.join(affordanceDict[noun])
    except TypeError:
        aff = None
    affVar.set(aff)

    index.set(nounSelected())
    lookUp()

# def searchEntry():
#     noun

# def updatestore() :
#     global storeList
#     newCat = [nounVar.get() + ":" + affVar.get()]
#     storeList += newCat
#     setSelect8 ()

def addMeaning(input=None) : 
    if affVar.get() != 'None':
        affVar.set(affVar.get() + "," +meaningList[meaningSelected ()]['synset'])
    else:
        affVar.set(meaningList[meaningSelected ()]['synset'])
    noun = affordanceDict.keys()[nounSelected()]
    addEntry()

# def addstore(input=None) : 
#     key = storeDict.keys()[storeSelected()]
#     try:
#         aff = ','.join(storeDict[key])
#     except TypeError:
#         aff = None
#     affVar.set(aff)
#     addEntry()


def nextEntry(input=None):
    noun = affordanceDict.keys()[index.get()+1]
    nounVar.set(noun)
    defVar.set(get_wn_meaning(noun))
    try:
        aff = ','.join(affordanceDict[noun])
    except TypeError:
        aff = None
    affVar.set(aff)
    index.set(index.get()+1)

def lastEntry(input=None):
    noun = affordanceDict.keys()[index.get()-1]
    nounVar.set(noun)
    defVar.set(get_wn_meaning(noun))
    try:
        aff = ','.join(affordanceDict[noun])
    except TypeError:
        aff = None
    affVar.set(aff)
    index.set(index.get()-1)

def saveDict(input=None):
    with open(affordance_dict_fpath, 'w') as f:
        f.write(json.dumps(affordanceDict))
    f.close()

    # with open(store_fpath, 'wb') as f2:
    #     pickle.dump(storeList, f2)
    # f2.close()

    msgVar.set("SAVED")


def makeWindow () :
    global nounVar, affVar, defVar, verbVar, verbMeanings, msgVar, select, select6, select8, index
    win = Tk()

    frame0 = Frame(win)       # select of names
    frame0.pack()
    scroll = Scrollbar(frame0, orient=VERTICAL)
    select = Listbox(frame0, yscrollcommand=scroll.set, height=10, width=50)
    scroll.config (command=select.yview)
    scroll.pack(side=RIGHT, fill=Y)
    select.pack(side=LEFT,  fill=BOTH, expand=1)
    select.bind('<Return>', loadEntry)
    index = IntVar()

    frame1 = Frame(win)
    frame1.pack()

    Label(frame1, text="Noun").grid(row=0, column=0, sticky=W)
    nounVar = StringVar()
    noun = Entry(frame1, textvariable=nounVar)
    noun.grid(row=0, column=1, sticky=W)

    Label(frame1, text="Definition").grid(row=1, column=0, sticky=W)
    defVar = StringVar()
    definition = Entry(frame1, textvariable=defVar, width=100)
    definition.grid(row=1, column=1, sticky=W)

    Label(frame1, text="WN tags").grid(row=2, column=0, sticky=W)
    affVar= StringVar()
    aff= Entry(frame1, textvariable=affVar, width=100)
    aff.grid(row=2, column=1, sticky=W)

    frame2 = Frame(win)       # Row of buttons
    frame2.pack()
    
    b1 = Button(frame2,text=" Add Entry ",command=addEntry)
    b1.pack(side=LEFT) 
    b2 = Button(frame2,text=" Load ",command=loadEntry)
    b2.pack(side=LEFT)
    # b3 = Button(frame2,text=" Search ", command=searchEntry)
    # b3.pack(side=LEFT)
    # b3 = Button(frame2,text=" Update store ",command=updatestore)
    # b3.pack(side=LEFT)

    frame3 = Frame(win)
    frame3.pack()
    Label(frame3, text="Press ENTER to load nounds; DOUBLE CLICK to add verb; COMMAND+a to save entry; COMMAND+n to load the next entry", bg='grey').grid(row=0, column=0, sticky=W)

    frame4 = Frame(win)
    frame4.pack()
    Label(frame4, text="Word to tagged").grid(row=0, column=0, sticky=W)
    verbVar = StringVar()
    verb = Entry(frame4, textvariable=verbVar)
    verb.grid(row=0, column=1, sticky=W)
    verb.bind('<Return>', lookUp)
    
    frame5 = Frame(win)
    frame5.pack()
    b4 = Button(frame5, text=" Look Up ",command=lookUp)
    b4.pack(side=LEFT)
    b5 = Button(frame5, text=" Add Tags ", command=addMeaning)
    b5.pack(side=LEFT)
    b6 = Button(frame5, text=" Save ", command=saveDict)
    b6.pack(side=LEFT)
    
    frame7 = Frame(win)
    frame7.pack()
    msgVar = StringVar()
    Label(frame7, textvariable=msgVar, fg='blue').grid(row=0, column=0, sticky=W)
    

    frame6 = Frame(win)
    frame6.pack()
    scroll6 = Scrollbar(frame6, orient = VERTICAL)
    select6 = Listbox(frame6, yscrollcommand=scroll6.set, height=10, width=120)
    scroll6.config (command=select6.yview)
    scroll6.pack(side=RIGHT, fill=Y)
    select6.pack(side=LEFT,  fill=BOTH, expand=1)
    select6.bind('<Double-Button>', addMeaning)
    
    # frame8 = Frame(win)
    # frame8.pack()
    # scroll8 = Scrollbar(frame8, orient = VERTICAL)
    # select8 = Listbox(frame8, yscrollcommand=scroll8.set, height=15, width=120)
    # scroll8.config (command=select8.yview)
    # scroll8.pack(side=RIGHT, fill=Y)
    # select8.pack(side=LEFT,  fill=BOTH, expand=1)
    # select8.bind('<Double-Button>', addstore)
    

    return win

def setSelect () :
    select.delete(0,END)
    for noun in affordanceDict:
        select.insert (END, noun)
    print index.get()
    select.selection_set(index.get())
    select.activate(index.get())


def setSelect6 ():
	select6.delete(0,END)
	for m in verbMeanings:
		select6.insert (END, m)

# def setSelect8 ():
#     select8.delete(0,END)
#     for c in storeList:
#         select8.insert (END, c)

win = makeWindow()
win.bind('<Command-s>', saveDict) 
win.bind('<Command-a>', addEntry)
win.bind('<Command-n>', nextEntry)
win.bind('<Command-b>', lastEntry)

setSelect ()
setSelect6 ()
# setSelect8 ()
win.mainloop()
