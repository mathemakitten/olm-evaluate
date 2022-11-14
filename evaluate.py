# Evaluation tasks supported: SuperGLUE, GDELT, CIA World Factbook, Whisper-created holdout
# SuperGLUE in run_superglue.py

""" SuperGLUE finetuning + evaluation is run on a separate script, run_superglue_.py """

"""Run evaluation on GDELT
# TODO: filter for reliable media
naive approach: feed in all news headlines, calculate perplexity, return average perplexity
"""
from gdelt.run_gdelt_evaluation import GdeltEvaluation

gd = GdeltEvaluation()

print('ugh')


"""Run evaluation on CIA World Factbook latest scrape via pseudologlikelihood
naive simple approach: feed in articles month-to-month; smarter would be to break up at a paragraph level and then
average the results over a document, because sequence length is 512. naive solution is to slice off the document from 
the beginning to 512, lol.
"""

""" TODO: Whisper-generated news data """