#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
import json
import codecs
from mult_w2v.multilingual_w2v import MultilingualW2V


# test a class which can train a W matrix to map between two word2vec indices
class MultilingualW2VTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(os.path.realpath(__file__))
        lang_1_code = 'en'
        lang_2_code = 'de'
        # 100 dim vectors
        lang_1_index = os.path.join(module_path, 'test_data/news-commentary-v8.en.w2v-model.bin')
        # 100 dim vectors
        lang_2_index = os.path.join(module_path, 'test_data/news-commentary-v8.de.w2v-model.bin')
        # 50 dim vectors (to test different vector sizes)
        lang_2_small = os.path.join(module_path, 'test_data/news-commentary-v8.de.w2v-model.50.bin')

        # load the word pairs
        word_pair_json = os.path.join(module_path, 'test_data/en-de.word-pairs.json')
        with codecs.open(word_pair_json, encoding='utf8') as json_in:
            bilingual_word_pairs = json.loads(json_in.read())

        self.mult_w2v = MultilingualW2V(lang_1_code, lang_2_code, lang_1_index, lang_2_index, bilingual_word_pairs)
        self.mult_w2v_different_dims = MultilingualW2V(lang_1_code, lang_2_code, lang_1_index, lang_2_small, bilingual_word_pairs)

    def test_train_index(self):
        self.assertTrue(self.mult_w2v.W is None)

        self.mult_w2v.train()
        self.assertTrue(self.mult_w2v.W is not None)

    # test that we learned something
    def test_bilingual_mapping(self):
        self.mult_w2v.train()

        source_word = u'mother'
        expected_target_word = u'Mutter'
        target_words_with_scores = self.mult_w2v.multilingual_most_similar(source_word)
        just_target_words = [u[0] for u in target_words_with_scores]
        self.assertTrue(expected_target_word in set(just_target_words))

    def test_different_dims(self):
        self.mult_w2v_different_dims.train()

        self.assertEqual(self.mult_w2v_different_dims.W.shape, (50, 100))

        source_word = u'father'
        expected_target_word = u'Vater'
        target_words_with_scores = self.mult_w2v_different_dims.multilingual_most_similar(source_word)
        just_target_words = [u[0] for u in target_words_with_scores]
        self.assertTrue(expected_target_word in set(just_target_words))

if __name__ == '__main__':
    unittest.main()

