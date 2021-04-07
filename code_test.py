# def ceiling_division(n, d):
#     return -(n // -d)

# def get_split(text, max_length=200, overlap=50):
#   l_total = []
#   l_parcial = []
#   text_len = len(text.split())
#   aux_value = (max_length - overlap)
#   splits = ceiling_division(text_len,aux_value)
#   if splits > 0:
#     n = splits
#   else: 
#     n = 1
#   for w in range(n):
#     if w == 0:
#       l_parcial = text.split()[:max_length]
#       l_total.append(" ".join(l_parcial))
#     else:
#       l_parcial = text.split()[w*aux_value:w*aux_value + max_length]
#       l_total.append(" ".join(l_parcial))
# #   import pdb; pdb.set_trace()
#   return l_total

text = 'O cuidado em identificar pontos críticos no desafiador cenário globalizado talvez venha a ressaltar a relatividade das formas de ação. A nível organizacional, a execução dos pontos do programa deve passar por modificações independentemente das diretrizes de desenvolvimento para o futuro. É importante questionar o quanto a determinação clara de objetivos desafia a capacidade de equalização do sistema de participação geral.\
        No entanto, não podemos esquecer que a estrutura atual da organização ainda não demonstrou convincentemente que vai participar na mudança dos paradigmas corporativos. Do mesmo modo, a competitividade nas transações comerciais garante a contribuição de um grupo importante na determinação do processo de comunicação como um todo. A prática cotidiana prova que a hegemonia do ambiente político nos obriga à análise do retorno esperado a longo prazo.\
        Percebemos, cada vez mais, que a constante divulgação das informações facilita a criação do sistema de formação de quadros que corresponde às necessidades. A certificação de metodologias que nos auxiliam a lidar com a contínua expansão de nossa atividade obstaculiza a apreciação da importância dos procedimentos normalmente adotados. As experiências acumuladas demonstram que a consulta aos diversos militantes assume importantes posições no estabelecimento dos índices pretendidos. É claro que o início da atividade geral de formação de atitudes é uma das consequências das condições financeiras e administrativas exigidas.'
# print(get_split(text))        

import gensim
model = gensim.models.doc2vec.Doc2Vec.load('models/itd_doc2vec_model')
v = model.infer_vector(text.split())
print(v)