from scipy.misc import imread
from nltk.corpus import wordnet
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt


def get_transcripts(url, txtname):
    import urllib.request
    from lxml import etree
    filename = './txts/' + url[35:-1] + '.txt'  # 提取季数和集数作为单独储存的文件名
    response = urllib.request.urlopen(url)
    xml = response.read()
    html = etree.HTML(xml)
    html_data = html.xpath('//p/span/text()')  # 根据台词信息的特点获取所有台词：所有p标签下的span标签中的内容
    if len(html_data) < 10:  # 如果上一种方法提取失败，就用另一种方法提取：所有div标签下的p标签中的内容
        html_data = html.xpath('//div/p/text()')
    with open(filename,'w', encoding='UTF-8')as f:  # 保存到分文档中，换行便于阅读
        f.writelines([line+'\n' for line in html_data])
    with open(txtname, 'a', encoding='UTF-8')as f:  # 保存到总文档中
        f.writelines([line+' ' for line in html_data])


def get_urls(sitemap, txtname):
    import urllib.request
    import xml.etree.ElementTree as ET
    response = urllib.request.urlopen(sitemap)
    xml = response.read()
    root = ET.fromstring(xml)  # 从parse中读出来的是tree,从fromstring中读出来的是root
    urls = [child[0].text for child in root]  # 遍历root的子树并提取第一个子节点的文本内容
    urls = urls[0:-3]  # 最后三个不是剧集的台词页面，而是网站的介绍页面,因此去掉
    for url in urls:
        get_transcripts(url, txtname)


def word_replace(new_text):
    import re
    # to find the punctuation
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_letter.sub(' ', new_text)
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def word_analyze(text):
    import nltk
    text_replace = word_replace(text)
    text_token = nltk.word_tokenize(text_replace)  # 分词

    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')  # 提取停词库
    text_stop = [word for word in text_token if len(word.lower())>1 and (word.lower()not in stop_words)]  # 去除停词
    taggled = nltk.pos_tag(text_stop)  # 滤除姓名数字等
    text_filtered = [word[0] for word in taggled if word[1] != 'NNP' and word[1] != 'CD']

    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    text_tag = nltk.pos_tag(text_filtered)  # 获取词性
    text_lemmatized = [wnl.lemmatize(tag[0], pos=get_wordnet_pos(tag[1]) or wordnet.NOUN) for tag in text_tag]

    fdist = nltk.FreqDist(text_lemmatized)
    # fdist = nltk.FreqDist(nltk.bigrams(text_lemmatized))  # 双词统计分析
    # fdist = nltk.FreqDist(nltk.trigrams(text_lemmatized))  # 三词统计分析
    # fdist.plot(20, cumulative=True)  # 频率分布图
    # fdist.tabulate(20)  # 频率分布表
    # fdist.tabulate(20, cumulative=True)  # 频率累计表

    fdist4wordcloud = nltk.FreqDist(text_lemmatized)
    return fdist4wordcloud


def draw_wordcloud(txtname, imgname, eptname, use_nltk):
    with open(txtname,'r', encoding='UTF-8') as f:
        mytext = f.read()
    back_coloring = imread(imgname)
    image_colors = ImageColorGenerator(back_coloring)
    wc = WordCloud(background_color="white",  # 背景颜色
                   max_words=2000,  # 词云显示的最大词数
                   mask=back_coloring,  # 设置背景图片
                   random_state=2,
                   margin=2  # 词语边缘像素距离
                   )
    if use_nltk:
        items = word_analyze(mytext)  # 进行分词等
        wc.generate_from_frequencies(items)
    else:
        wc.generate(mytext)
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.show()
    wc.to_file(eptname)


if __name__ == '__main__':
    use_separate = False  # 是否使用手动分词
    use_crawler = False  # 是否重新爬取台词
    sitemap = 'https://bigbangtrans.wordpress.com/sitemap.xml'  # sitemap地址
    imgname = "./pics/templet.jpg"  # 作为绘制词云的模板图片
    txtname = "./txts/transcripts.txt"  # 台词爬取保存的文本文件
    eptname = "./pics/export.jpg"  # 词云绘制的保存路径
    if use_crawler:
        get_urls(sitemap, txtname)
    draw_wordcloud(txtname, imgname, eptname, use_separate)
