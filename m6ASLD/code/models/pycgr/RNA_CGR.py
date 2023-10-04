""""""
"""
Chaos Game Representation
将蛋白质序列转为CGR序列，蛋白质序列为原来长度的三倍
同时将爬虫读取的ID-pssm概率加到对应位置
"""
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
from .helper import *

from Bio import SeqIO

# defining cgr graph
# CGR_CENTER = (0.5, 0.5)
CGR_X_MAX = 1
CGR_Y_MAX = 1
CGR_X_MIN = 0
CGR_Y_MIN = 0
CGR_A = (CGR_X_MIN, CGR_Y_MIN) #为ATCG设置顶点
CGR_T = (CGR_X_MAX, CGR_Y_MIN)
CGR_G = (CGR_X_MAX, CGR_Y_MAX)
CGR_C = (CGR_X_MIN, CGR_Y_MAX)
CGR_CENTER = ((CGR_X_MAX - CGR_Y_MIN) / 2, (CGR_Y_MAX - CGR_Y_MIN) / 2) #设置ATCG中心

# read and store
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def empty_dict():
	"""
	None type return vessel for defaultdict
	:return:
	"""
	return None


CGR_DICT = defaultdict(
	empty_dict,
	[
		('A', CGR_A),  # Adenine
		('T', CGR_T),  # Thymine
		('G', CGR_G),  # Guanine
		('C', CGR_C),  # Cytosine
		('U', CGR_T),  # Uracil demethylated form of thymine
		('a', CGR_A),  # Adenine
		('t', CGR_T),  # Thymine
		('g', CGR_G),  # Guanine
		('c', CGR_C),  # Cytosine
		('u', CGR_T)  # Uracil/Thymine
	]
)


# yield的函数可返回一个可迭代的的generator对象，可以用for循环或者调用next()方法遍历生成器对象来提取结果。
def fasta_reader(fasta): #读取fasta数据
	"""Return a generator with sequence description and sequence

	:param fasta: str filename
	"""
	# TODO: modify it to be capable of reading genebank etc
	flist = SeqIO.parse(fasta, "fasta")
	for i in flist:
		yield i.description, i.seq


def mk_cgr(seq): #获得序列中每个碱基对应的坐标（二重列表）
	"""Generate cgr

	:param seq: list of nucleotide
	:return cgr: [['nt', (x, y)]] List[List[Tuple(float, float)]]
	"""
	cgr = []
	cgr_marker = CGR_CENTER[:
		]    # The center of square which serves as first marker
	for s in seq:
		cgr_corner = CGR_DICT[s]
		if cgr_corner:
			cgr_marker = (
				(cgr_corner[0] + cgr_marker[0]) / 2,
				(cgr_corner[1] + cgr_marker[1]) / 2
			)
			cgr.append([s, cgr_marker])
		else:
			sys.stderr.write("Bad Nucleotide: " + s + " \n")

	return cgr


def mk_plot(cgr, name, figid): #绘制CGR图
	"""Plotting the cgr
		:param cgr: [(A, (0.1, 0.1))]
		:param name: str
		:param figid: int
		:return dict: {'fignum': figid, 'title': name, 'fname': helper.slugify(name)}
	"""
	x_axis = [i[1][0] for i in cgr]
	y_axis = [i[1][1] for i in cgr]
	plt.figure(figid)
	plt.title("Chaos Game Representation\n" + name, wrap=True)
	# diagonal and vertical cross
	# plt.plot([x1, x2], [y1, y2])
	# plt.plot([0.5,0.5], [0,1], 'k-')
	plt.plot([CGR_CENTER[0], CGR_CENTER[0]], [0, CGR_Y_MAX], 'k-')

	# plt.plot([0,1], [0.5,0.5], 'k-')
	plt.plot([CGR_Y_MIN, CGR_X_MAX], [CGR_CENTER[1], CGR_CENTER[1]], 'k-')
	plt.scatter(x_axis, y_axis, alpha=0.5, marker='.')

	return {'fignum': figid, 'title': name, 'fname': slugify(name)}


def write_figure(fig, output_dir, dpi=300): #图片保存
	"""Write plot to png
	:param fig:  {'fignum':figid, 'title':name, 'fname':helper.slugify(name)}
	:param dpi: int dpi of output
	:param output_dir: str

	Usage:
		figures = [mk_plot(cgr) for cgr in all_cgr]
		for fig in figures:
			write_figure(fig, "/var/tmp/")
		The figid in the mk_plot's return dict must be present in plt.get_fignums()
	"""
	all_figid = plt.get_fignums()
	if fig['fignum'] not in all_figid:
		raise ValueError("Figure %i not present in figlist" % fig['fignum'])
	plt.figure(fig['fignum'])
	target_name = os.path.join(
		output_dir,
		slugify(fig['fname']) + ".png"
	)
	plt.savefig(target_name, dpi=dpi)


def RNA_CGR1(RNA_FILE):
	"""
	对同一氨基酸位置的内在无序概率和保守概率进行加权求和，得到拉普拉斯值
	"""
	# ID_pro_1783_all = [] #内在无序概率
	# pssm_pro_1783_max_min = [] #保守概率
	#
	# ReadMyCsv(ID_pro_1783_all, 'ID_pro_1783_all.csv')
	# ReadMyCsv(pssm_pro_1783_max_min, 'pssm_pro_1783_max_min.csv')

	# # print(ID_pro_1785_all[89])
	# list1=[i for i in ID_pro_1783_all[77] if i !='']
	# print(len(list1))
	# # print(ID_pro_1785_all[89][1])
	# print(len(pssm_pro_1783_max_min[77]))

	# k=0
	# ID_pssm=[]
	# while k<len(ID_pro_1783_all):
	#     g=0
	#     temp=[]
	#     print(k)
	#     print(len(pssm_pro_1783_max_min[k]))
	#     while g<len(pssm_pro_1783_max_min[k]):
	#         # print(ID_pro_1785_all[k])
	#         mean=(float(ID_pro_1783_all[k][g])+float(pssm_pro_1783_max_min[k][g]))/2
	#         temp.append(mean)
	#         g+=1
	#     ID_pssm.append(temp)
	#     k+=1
	# StorFile(ID_pssm,'ID_pssm_prob.csv')

	# fig_id = 1
	for ele in RNA_FILE:
		seq1=ele[1] #将多肽氨基酸序列转换为DNA序列
		cgr1 = mk_cgr(seq1) #cgr表示一条序列的
		# print('name len(cgr1)'+str(ele[0])+ ' ' +str(len(cgr1)))
		Temp_CGR_Seq_Pep = []
		#使用弧度来表示该CGR序列
		for elel in cgr1: #构造CGR游走序列(选择使用角度来表示游走序列的元素 )
			# Temp_CGR_Seq_Pep.append(elel[1][1]-elel[1][0]) #转换为弧度
			Temp_CGR_Seq_Pep.append(math.atan2(elel[1][1],elel[1][0])) #转换为弧度
		# fig_id += 1
		# print(fig_id)
		ele[1]=Temp_CGR_Seq_Pep

	return RNA_FILE


def RNA_CGR2(RNA_FILE):
	"""
	对同一氨基酸位置的内在无序概率和保守概率进行加权求和，得到拉普拉斯值
	"""
	# ID_pro_1783_all = [] #内在无序概率
	# pssm_pro_1783_max_min = [] #保守概率
	#
	# ReadMyCsv(ID_pro_1783_all, 'ID_pro_1783_all.csv')
	# ReadMyCsv(pssm_pro_1783_max_min, 'pssm_pro_1783_max_min.csv')

	# # print(ID_pro_1785_all[89])
	# list1=[i for i in ID_pro_1783_all[77] if i !='']
	# print(len(list1))
	# # print(ID_pro_1785_all[89][1])
	# print(len(pssm_pro_1783_max_min[77]))

	# k=0
	# ID_pssm=[]
	# while k<len(ID_pro_1783_all):
	#     g=0
	#     temp=[]
	#     print(k)
	#     print(len(pssm_pro_1783_max_min[k]))
	#     while g<len(pssm_pro_1783_max_min[k]):
	#         # print(ID_pro_1785_all[k])
	#         mean=(float(ID_pro_1783_all[k][g])+float(pssm_pro_1783_max_min[k][g]))/2
	#         temp.append(mean)
	#         g+=1
	#     ID_pssm.append(temp)
	#     k+=1
	# StorFile(ID_pssm,'ID_pssm_prob.csv')

	# fig_id = 1
	for ele in RNA_FILE:
		seq1 = ele[1]  # 将多肽氨基酸序列转换为DNA序列
		print('seq1 '+str(seq1))
		cgr1 = mk_cgr(seq1)  # cgr表示一条序列的
		# print('name len(cgr1)' + str(ele[0]) + ' ' + str(len(cgr1)))
		Temp_CGR_Seq_Pep = []
		# 使用弧度来表示该CGR序列
		for elel in cgr1:  # 构造CGR游走序列(选择使用角度来表示游走序列的元素 )
			Temp_CGR_Seq_Pep.append(math.atan2(elel[1][1], elel[1][0]))  # 转换为弧度
		# fig_id += 1
		# print(fig_id)
		ele[1] = Temp_CGR_Seq_Pep

	return RNA_FILE

	# if args.save:
	# 	for i in my_plots:
	# 		write_figure(i, args.save_dir, dpi=args.dpi)
	#
	# if args.show:
	# 	plt.show()

