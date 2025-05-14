# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from transformers.commands import train

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


class MnliProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()
        self.labels = ["contradiction", "entailment", "neutral"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label) - 1)
                examples.append(example)

        return examples


class AgnewsProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "agnews"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["World", "Sports", "Business", "Tech"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label) - 1)
                examples.append(example)
        return examples


class DBpediaProcessor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "dbpedia"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 14
        assert len(trainvalid_dataset) == 560000
        assert len(test_dataset) == 70000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["company", "school", "artist", "athlete", "politics", "transportation", "building", "river", "village", "animal", "plant",
                       "album", "film", "book", ]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)), 'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a + "."
                text_b = ". ".join(text_b)
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(labels[idx]))
                examples.append(example)
        return examples


class ImdbProcessor(DataProcessor):
    """
    `IMDB <https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf>`_ is a Movie Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "imdb"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert len(trainvalid_dataset) == 25000
        assert len(test_dataset) == 25000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)), 'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples

    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file = open(os.path.join(data_dir, dirname, "{}_labels.txt".format('test')), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        return labels


# class AmazonProcessor(DataProcessor):
#     """
#     `Amazon <https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf>`_ is a Product Review Sentiment Classification dataset.

#     we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

#     Examples: # TODO implement this
#     """

#     def __init__(self):
#         # raise NotImplementedError
#         super().__init__()
#         self.labels = ["bad", "good"]

#     def get_examples(self, data_dir, split):
#         examples = []
#         label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
#         labels = [int(x.strip()) for x in label_file.readlines()]
#         if split == "test":
#             logger.info("Sample a mid-size test set for effeciecy, use sampled_test_idx.txt")
#             with open(os.path.join(self.args.data_dir,self.dirname,"sampled_test_idx.txt"),'r') as sampleidxfile:
#                 sampled_idx = sampleidxfile.readline()
#                 sampled_idx = sampled_idx.split()
#                 sampled_idx = set([int(x) for x in sampled_idx])

#         with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
#             for idx, line in enumerate(fin):
#                 if split=='test':
#                     if idx not in sampled_idx:
#                         continue
#                 text_a = line.strip()
#                 example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
#                 examples.append(example)
#         return examples


class AmazonProcessor(DataProcessor):
    """
    `Amazon <https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf>`_ is a Product Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples: # TODO implement this
    """

    def __init__(self):
        super().__init__()
        self.labels = ["bad", "good"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)), 'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples


class YahooProcessor(DataProcessor):
    """
    Yahoo! Answers Topic Classification Dataset

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports",
                       "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '), question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label) - 1)
                examples.append(example)
        return examples


# class SST2Processor(DataProcessor):
#     """
#     #TODO test needed
#     """

#     def __init__(self):
#         raise NotImplementedError
#         super().__init__()
#         self.labels = ["negative", "positive"]

#     def get_examples(self, data_dir, split):
#         examples = []
#         path = os.path.join(data_dir,"{}.tsv".format(split))
#         with open(path, 'r') as f:
#             reader = csv.DictReader(f, delimiter='\t')
#             for idx, example_json in enumerate(reader):
#                 text_a = example_json['sentence'].strip()
#                 example = InputExample(guid=str(idx), text_a=text_a, label=int(example_json['label']))
#                 examples.append(example)
#         return examples
class SST2Processor(DataProcessor):
    """
    `SST-2 <https://nlp.stanford.edu/sentiment/index.html>`_ dataset is a dataset for sentiment analysis. It is a modified version containing only binary labels (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) on top of the original 5-labeled dataset released first in `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank <https://aclanthology.org/D13-1170.pdf>`_

    We use the data released in `Making Pre-trained Language Models Better Few-shot Learners (Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.lmbff_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SST-2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert processor.get_labels() == ['0','1']
        assert len(train_dataset) == 6920
        assert len(dev_dataset) == 872
        assert len(test_dataset) == 1821
        assert train_dataset[0].text_a == 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films'
        assert train_dataset[0].label == 1

    """

    def __init__(self):
        super().__init__()
        self.labels = ['0', '1']

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=self.get_label_id(label))
                examples.append(example)
        return examples


class GIDProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        # 此处的label使用原生的即可，不用处理，暂时没有添加O标签。
        self.labels = []

    def get_examples(self, data_dir, split, **kwargs):
        # 需要一个传参，来判断是pretrain阶段还是discover阶段
        # train 与 dev 的ind标签保持不变，但ood标签变更为ood，test的ind与ood都维持原样
        # 针对GID基础数据集，将dev改名为eval，其余名称不变
        split = 'eval' if split == 'dev' else split
        path_list = []

        if kwargs.get('mode') == 'pretrain':
            path_for_ind = os.path.join(data_dir, "{}_ind.csv".format(split))
            path_list = [path_for_ind]

        elif kwargs.get('mode') == 'discover' or kwargs.get('mode') == 'detect':

            path_list = []
            path_for_ind = os.path.join(data_dir, "{}_ind.csv".format(split))
            path_for_ood = os.path.join(data_dir, "{}_ood.csv".format(split))
            if os.path.exists(path_for_ind):
                path_list.append(path_for_ind)
            if os.path.exists(path_for_ood):
                path_list.append(path_for_ood)

        # todo 增加一个mode为detect，二分类，检测为ind还是ood
        examples = []
        for idx_path, path in enumerate(path_list):
            with open(path, encoding='utf8') as f:
                reader = csv.reader(f, delimiter='\t')
                for idx, row in enumerate(reader):
                    if idx == 0:
                        continue
                    body, label = row
                    text_a = body.strip()

                    if idx_path == 0:
                        label = label.strip()

                    # todo 暂时把验证集的ood标签也放出来，后面需要改进
                    if kwargs.get('mode') == 'discover':
                        if idx_path == 1 and split in ['train']:
                            label = 'OOD'
                        if idx_path == 1 and split == 'test':
                            label = label.strip()
                    if kwargs.get('mode') == 'detect':
                        label = 'IND' if idx_path == 0 else 'OOD'

                    # # train 与 dev 的ind标签保持不变，但ood标签变更为ood，test的ind与ood都维持原样
                    # # todo 这里可能不是直接打上ood，而是使用gpt来提前打上伪标签，附在文件的第三列，使用伪标签。
                    # if idx_path == 1 and split != 'test':
                    #     label = 'OOD'
                    example = InputExample(guid=str(idx), text_a=text_a, label=self.get_label_id(label))
                    examples.append(example)
        return examples


class GID_SDProcessor(GIDProcessor):
    def __init__(self):
        super().__init__()
        # 此处的label使用原生的即可，不用处理，暂时没有添加O标签。
        self.labels = ['age_limit', 'card_swallowed', 'terminate_account', 'beneficiary_not_allowed', 'failed_transfer',
                       'card_payment_not_recognised', 'cash_withdrawal_charge', 'change_pin', 'pending_cash_withdrawal', 'transfer_timing',
                       'wrong_exchange_rate_for_cash_withdrawal', 'transfer_not_received_by_recipient',
                       'balance_not_updated_after_cheque_or_cash_deposit', 'top_up_by_bank_transfer_charge', 'verify_source_of_funds',
                       'pending_top_up', 'declined_transfer', 'pending_card_payment', 'transfer_fee_charged', 'exchange_charge', 'card_linking',
                       'top_up_by_cash_or_cheque', 'apple_pay_or_google_pay', 'wrong_amount_of_cash_received', 'reverted_card_payment?',
                       'verify_my_identity', 'visa_or_mastercard', 'get_physical_card', 'direct_debit_payment_not_recognised',
                       'unable_to_verify_identity', 'edit_personal_details', 'lost_or_stolen_card', 'country_support', 'lost_or_stolen_phone',
                       'supported_cards_and_currencies', 'top_up_reverted', 'declined_card_payment', 'automatic_top_up', 'pin_blocked',
                       'why_verify_identity', 'verify_top_up', 'card_arrival', 'card_delivery_estimate', 'compromised_card', 'passcode_forgotten',
                       'card_acceptance', 'getting_virtual_card', 'activate_my_card', 'fiat_currency_support', 'topping_up_by_card',
                       'balance_not_updated_after_bank_transfer', 'cancel_transfer', 'exchange_rate', 'disposable_card_limits', 'exchange_via_app',
                       'cash_withdrawal_not_recognised', 'transfer_into_account', 'transaction_charged_twice', 'card_payment_wrong_exchange_rate',
                       'get_disposable_virtual_card', 'contactless_not_working', 'top_up_limits', 'card_about_to_expire', 'top_up_by_card_charge',
                       'extra_charge_on_statement', 'order_physical_card', 'top_up_failed', 'getting_spare_card', 'card_payment_fee_charged',
                       'card_not_working', 'virtual_card_not_working', 'atm_support', 'request_refund', 'declined_cash_withdrawal',
                       'Refund_not_showing_up', 'pending_transfer', 'receiving_money', 'OOD']


class GID_MDProcessor(GIDProcessor):
    def __init__(self):
        super().__init__()
        # 此处的label使用原生的即可，不用处理，暂时没有添加O标签。
        self.labels = ['payday', 'balance', 'rewards_balance', 'definition', 'weather', 'measurement_conversion', 'ingredient_substitution',
                       'bill_balance', 'direct_deposit', 'goodbye', 'todo_list', 'calories', 'change_user_name', 'yes', 'who_made_you',
                       'restaurant_reviews', 'text', 'mpg', 'account_blocked', 'w2', 'pay_bill', 'repeat', 'cancel_reservation', 'translate',
                       'shopping_list_update', 'carry_on', 'roll_dice', 'who_do_you_work_for', 'traffic', 'reset_settings', 'time', 'credit_score',
                       'how_old_are_you', 'make_call', 'meeting_schedule', 'freeze_account', 'next_song', 'thank_you', 'expiration_date',
                       'rollover_401k', 'card_declined', 'apr', 'distance', 'new_card', 'what_are_your_hobbies', 'change_speed', 'no',
                       'what_is_your_name', 'car_rental', 'insurance', 'maybe', 'where_are_you_from', 'tire_change', 'min_payment',
                       'schedule_meeting', 'oil_change_how', 'greeting', 'tell_joke', 'ingredients_list', 'current_location', 'order', 'transactions',
                       'calendar_update', 'exchange_rate', 'lost_luggage', 'oil_change_when', 'are_you_a_bot', 'travel_suggestion', 'calculator',
                       'shopping_list', 'travel_notification', 'whisper_mode', 'change_ai_name', 'play_music', 'sync_device', 'book_flight',
                       'next_holiday', 'calendar', 'international_visa', 'what_song', 'directions', 'bill_due', 'reminder_update', 'food_last',
                       'replacement_card_duration', 'pto_request_status', 'meaning_of_life', 'timezone', 'pin_change', 'schedule_maintenance',
                       'user_name', 'plug_type', 'interest_rate', 'flip_coin', 'spending_history', 'cook_time', 'fun_fact', 'flight_status',
                       'share_location', 'timer', 'how_busy', 'travel_alert', 'spelling', 'pto_used', 'smart_home', 'credit_limit_change',
                       'accept_reservations', 'reminder', 'what_can_i_ask_you', 'pto_balance', 'tire_pressure', 'meal_suggestion', 'damaged_card',
                       'change_accent', 'recipe', 'restaurant_reservation', 'credit_limit', 'taxes', 'nutrition_info', 'vaccines',
                       'restaurant_suggestion', 'jump_start', 'alarm', 'order_checks', 'gas', 'pto_request', 'date', 'update_playlist',
                       'todo_list_update', 'income', 'report_fraud', 'international_fees', 'transfer', 'change_language', 'confirm_reservation',
                       'book_hotel', 'report_lost_card', 'uber', 'gas_type', 'insurance_change', 'order_status', 'change_volume', 'cancel',
                       'application_status', 'find_phone', 'redeem_rewards', 'do_you_have_pets', 'last_maintenance', 'routing',
                       'improve_credit_score', 'OOD']

class GID_CDProcessor(GIDProcessor):
    def __init__(self):
        super().__init__()
        # 此处的label使用原生的即可，不用处理，暂时没有添加O标签。
        self.labels = ['payday', 'balance', 'rewards_balance', 'definition', 'weather', 'measurement_conversion', 'ingredient_substitution',
                       'bill_balance', 'direct_deposit', 'goodbye', 'todo_list', 'calories', 'change_user_name', 'yes', 'who_made_you',
                       'restaurant_reviews', 'text', 'mpg', 'account_blocked', 'w2', 'pay_bill', 'repeat', 'cancel_reservation', 'translate',
                       'shopping_list_update', 'carry_on', 'roll_dice', 'who_do_you_work_for', 'traffic', 'reset_settings', 'time', 'credit_score',
                       'how_old_are_you', 'make_call', 'meeting_schedule', 'freeze_account', 'next_song', 'thank_you', 'expiration_date',
                       'rollover_401k', 'card_declined', 'apr', 'distance', 'new_card', 'what_are_your_hobbies', 'change_speed', 'no',
                       'what_is_your_name', 'car_rental', 'insurance', 'maybe', 'where_are_you_from', 'tire_change', 'min_payment',
                       'schedule_meeting', 'oil_change_how', 'greeting', 'tell_joke', 'ingredients_list', 'current_location', 'order', 'transactions',
                       'calendar_update', 'exchange_rate', 'lost_luggage', 'oil_change_when', 'are_you_a_bot', 'travel_suggestion', 'calculator',
                       'shopping_list', 'travel_notification', 'whisper_mode', 'change_ai_name', 'play_music', 'sync_device', 'book_flight',
                       'next_holiday', 'calendar', 'international_visa', 'what_song', 'directions', 'bill_due', 'reminder_update', 'food_last',
                       'replacement_card_duration', 'pto_request_status', 'meaning_of_life', 'timezone', 'pin_change', 'schedule_maintenance',
                       'user_name', 'plug_type', 'interest_rate', 'flip_coin', 'spending_history', 'cook_time', 'fun_fact', 'flight_status',
                       'share_location', 'timer', 'how_busy', 'travel_alert', 'spelling', 'pto_used', 'smart_home', 'credit_limit_change',
                       'accept_reservations', 'reminder', 'what_can_i_ask_you', 'pto_balance', 'tire_pressure', 'meal_suggestion', 'damaged_card',
                       'change_accent', 'recipe', 'restaurant_reservation', 'credit_limit', 'taxes', 'nutrition_info', 'vaccines',
                       'restaurant_suggestion', 'jump_start', 'alarm', 'order_checks', 'gas', 'pto_request', 'date', 'update_playlist',
                       'todo_list_update', 'income', 'report_fraud', 'international_fees', 'transfer', 'change_language', 'confirm_reservation',
                       'book_hotel', 'report_lost_card', 'uber', 'gas_type', 'insurance_change', 'order_status', 'change_volume', 'cancel',
                       'application_status', 'find_phone', 'redeem_rewards', 'do_you_have_pets', 'last_maintenance', 'routing',
                       'improve_credit_score', 'OOD']


class GID_IND_OODProcessor(GIDProcessor):
    def __init__(self):
        super().__init__()
        # 此处的label使用原生的即可，不用处理，暂时没有添加O标签。
        self.labels = ['IND', 'OOD']

PROCESSORS = {
    "agnews": AgnewsProcessor,
    "dbpedia": DBpediaProcessor,
    "amazon": AmazonProcessor,
    "imdb": ImdbProcessor,
    "sst-2": SST2Processor,
    "mnli": MnliProcessor,
    "yahoo": YahooProcessor,
    "gid-sd": GID_SDProcessor,
    "gid-md": GID_MDProcessor,
    "gid-cd": GID_CDProcessor,
    "gid-ind-ood": GID_IND_OODProcessor,
}
