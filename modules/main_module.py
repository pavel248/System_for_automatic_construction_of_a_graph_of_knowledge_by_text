from nltk.tokenize import sent_tokenize
import pandas as pd
import stanza
import pymorphy2


def norm_form(morph, word):
    return morph.parse(word)[0].normal_form


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_graph(text):
    morph = pymorphy2.MorphAnalyzer(lang="ru")

    stanza.download('ru')
    nlp = stanza.Pipeline('ru')
    nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,ner,depparse', use_gpu=True)

    full_corpus = text

    sentences = tokenize_text(full_corpus)

    print(sentences)

    triplets = get_triplets(nlp, sentences)
    print(triplets)

    for_df = get_triplets_for_df(triplets)

    print(for_df)

    gr_num, info_dict, label_dict, nodes, word_num = method_name(for_df, morph)

    draw_graph(gr_num, info_dict, label_dict, nodes, word_num)


def draw_graph(gr_num, info_dict, label_dict, nodes, word_num):
    import modules.script_for_graph
    import importlib
    importlib.reload(modules.script_for_graph)
    from modules.script_for_graph import header_text, tail_text
    header_text += """\nvar nodes = new vis.DataSet([\n"""
    for w in nodes:
        header_text += "{"
        header_text += f"""         id: {word_num[w]}, 
                                        label: "{w}"\n"""
        header_text += "},"
    header_text += "   ]);\n"
    header_text += """var edges = new vis.DataSet(["""
    for k in info_dict.keys():
        header_text += "{"
        header_text += f"""       from: {word_num[k[0]]}, 
                            to: {word_num[k[1]]}, 
                            arrows: "to",
                            label: "{label_dict[k]}",
                            info: {info_dict[k]}\n"""
        header_text += "},"
    header_text += "   ]);\n"
    full_text = ""
    full_text += header_text
    full_text += tail_text
    with open(f"templates\Graph_for_group_{gr_num}.html", "w", encoding="utf-8") as f:
        f.write(full_text)


def method_name(for_df, morph):
    df_triplets = pd.DataFrame(for_df, columns=["full_sent", "subject", "verb", "object"])
    df_triplets["subj_n_f"] = df_triplets["subject"].apply(lambda x: norm_form(morph, x))
    df_triplets["obj_n_f"] = df_triplets["object"].apply(lambda x: norm_form(morph, x))
    print(df_triplets.head(5))
    groups = list(chunks(df_triplets["obj_n_f"].unique(), 100))
    gr_num = 0
    df_for_draw = df_triplets[df_triplets["obj_n_f"].isin(groups[gr_num])]
    nodes = pd.unique(df_for_draw[["subj_n_f", "obj_n_f"]].values.ravel("K"))
    df_d_d = df_for_draw.drop_duplicates(subset=["subj_n_f", "obj_n_f", "verb"])[
        ["subj_n_f", "obj_n_f", "verb", "full_sent"]]
    info_dict = dict()
    label_dict = dict()
    for cc, raw in enumerate(df_d_d.values):
        info_dict[(raw[0], raw[1])] = {f"sent_{cc}": raw[3]}
        label_dict[(raw[0], raw[1])] = raw[2]
    word_num = dict()
    for c, word in enumerate(nodes):
        word_num[word] = c + 1
    return gr_num, info_dict, label_dict, nodes, word_num


def get_triplets_for_df(triplets):
    clear_triplets = dict()
    for tr in triplets:
        for k in tr[1].keys():
            if "obj" in tr[1][k].keys():
                clear_triplets[tr[0]] = [k, tr[1][k]['head'], tr[1][k]['obj']]
    for_df = []
    for k in clear_triplets.keys():
        for_df.append([k] + clear_triplets[k])
    return for_df


def get_triplets(nlp, sentences):
    triplets = []
    for s in sentences:
        doc = nlp(s)
        for sent in doc.sentences:
            entities = [ent.text for ent in sent.ents]
            res_d = dict()
            temp_d = dict()
            for word in sent.words:
                temp_d[word.text] = {"head": sent.words[word.head - 1].text, "dep": word.deprel, "id": word.id}
            for k in temp_d.keys():
                nmod_1 = ""
                nmod_2 = ""
                if (temp_d[k]["dep"] in ["nsubj", "nsubj:pass"]) & (k in entities):
                    res_d[k] = {"head": temp_d[k]["head"]}

                    for k_0 in temp_d.keys():
                        if (temp_d[k_0]["dep"] in ["obj", "obl"]) & \
                                (temp_d[k_0]["head"] == res_d[k]["head"]) & \
                                (temp_d[k_0]["id"] > temp_d[res_d[k]["head"]]["id"]):
                            res_d[k]["obj"] = k_0
                            break

                    for k_1 in temp_d.keys():
                        if (temp_d[k_1]["head"] == res_d[k]["head"]) & (k_1 == "не"):
                            res_d[k]["head"] = "не " + res_d[k]["head"]

                    if "obj" in res_d[k].keys():
                        for k_4 in temp_d.keys():
                            if (temp_d[k_4]["dep"] == "nmod") & \
                                    (temp_d[k_4]["head"] == res_d[k]["obj"]):
                                nmod_1 = k_4
                                break

                        for k_5 in temp_d.keys():
                            if (temp_d[k_5]["dep"] == "nummod") & \
                                    (temp_d[k_5]["head"] == nmod_1):
                                nmod_2 = k_5
                                break
                        if nmod_1 == "" and nmod_2 != "":
                            res_d[k]["obj"] = res_d[k]["obj"] + " " + nmod_2
                        if nmod_2 == "" and nmod_1 != "":
                            res_d[k]["obj"] = res_d[k]["obj"] + " " + nmod_1
                        if nmod_2 != "" and nmod_1 != "":
                            res_d[k]["obj"] = res_d[k]["obj"] + " " + nmod_2 + " " + nmod_1
                        if nmod_2 == "" and nmod_1 == "":
                            res_d[k]["obj"] = res_d[k]["obj"]

            if len(res_d) > 0:
                triplets.append([s, res_d])
    return triplets


def tokenize_text(full_corpus):
    try:
        sentences = [sent for sent in sent_tokenize(full_corpus, language="russian")]
    except:
        import nltk
        nltk.download('punkt')
        sentences = [sent for corp in full_corpus for sent in sent_tokenize(corp, language="russian")]
    return sentences
