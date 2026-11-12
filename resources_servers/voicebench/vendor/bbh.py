# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/MatthewCYM/VoiceBench/blob/0842845a532af34396b173f92f304aa9e2b3c4ab/src/evaluator/bbh.py
# Changes: remove unused dataset aggregation/dependencies, use request-local RNG,
# remove fallback console prints, and format. Parsing/check behavior is preserved.
# See PROVENANCE.json and LICENSE for source hashes and attribution.

import re

from . import randomness as random


class BBHEvaluator:
    def extract_answer(self, response, id):
        response = self.normalize(response)
        if "sports_understanding" in id:
            return self.extract_answer_sports(response)
        elif "hyperbaton" in id:
            return self.extract_answer_hyperbaton(response)
        elif "navigate" in id:
            return self.extract_answer_navigate(response)
        elif "web_of_lies" in id:
            return self.extract_answer_lies(response)
        else:
            raise NotImplementedError

    def normalize(self, response):
        response = response.lower()
        if response.endswith("<|user|>"):
            response = response[:-8].strip()
        if response.startswith("<1>") or response.startswith("<2>") or response.startswith("<3>"):
            response = response[3:].strip()
        response = response.replace("<|turn_end|>", "")
        response = (
            response.replace(":", " ")
            .replace("**", " ")
            .replace('"', " ")
            .replace("-", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace("：", " ")
        )
        response = " ".join(response.split())
        return response

    def extract_answer_hyperbaton(self, response):
        if response == "a":
            return 0
        elif response == "b":
            return 0
        elif "the answer is (a)" in response:
            return 0
        elif "the answer is (b)" in response:
            return 1
        elif "the correct adjective order is option (a)" in response:
            return 0
        elif "the correct adjective order is option (b)" in response:
            return 1
        elif "the correct grammatical order is a" in response:
            return 0
        elif "the correct grammatical order is b" in response:
            return 1
        elif "the correct sentence is option (a)" in response:
            return 0
        elif "the correct sentence is option (b)" in response:
            return 1
        elif "the sentence with the correct adjectives is a" in response:
            return 0
        elif "the sentence with the correct adjectives is b" in response:
            return 1
        elif "correct adjective order options is a" in response:
            return 0
        elif "correct adjective order options is b" in response:
            return 1
        elif "the correct answer is (a)" in response:
            return 0
        elif "the correct answer is (b)" in response:
            return 1
        elif "the correct option is (a)" in response:
            return 0
        elif "the correct option is (b)" in response:
            return 1
        elif "the correct order of adjectives in the sentence is option a" in response:
            return 0
        elif "the correct order of adjectives in the sentence is option b" in response:
            return 1
        elif "the correct sentence would be option (a)" in response:
            return 0
        elif "the correct sentence would be option (b)" in response:
            return 1
        elif "the correct sentence is (a)" in response:
            return 0
        elif "the correct sentence is (b)" in response:
            return 1
        elif "the proper adjective order is a" in response:
            return 0
        elif "the proper adjective order is b" in response:
            return 1
        elif "the correct adjective order is (a)" in response:
            return 0
        elif "the correct adjective order is (b)" in response:
            return 1
        elif "correct adjective order is a" in response:
            return 0
        elif "correct adjective order is b" in response:
            return 1
        elif "the adjectives in option a are in the correct order" in response:
            return 0
        elif "the adjectives in option b are in the correct order" in response:
            return 1
        elif "the right adjective order is a)" in response:
            return 0
        elif "the right adjective order is b)" in response:
            return 1
        elif "the correct adjectivs is (a)" in response:
            return 0
        elif "the correct adjectivs is (b)" in response:
            return 1
        elif "the right adjective order is (a)" in response:
            return 0
        elif "the right adjective order is (b)" in response:
            return 1
        elif "correct adjectival order is a" in response:
            return 0
        elif "correct adjectival order is b" in response:
            return 1
        elif "the proper adjective order is (a)" in response:
            return 0
        elif "the proper adjective order is (b)" in response:
            return 1
        elif "the correct sentence order is a" in response:
            return 0
        elif "the correct sentence order is b" in response:
            return 1
        elif "option a correctly follows the standard adjective order" in response:
            return 0
        elif "option b correctly follows the standard adjective order" in response:
            return 1
        elif "the answer directly is a" in response:
            return 0
        elif "the answer directly is b" in response:
            return 1
        elif "the correct order would be option a" in response:
            return 0
        elif "the correct order would be option b" in response:
            return 1
        elif "final answer should be a" in response:
            return 0
        elif "final answer should be b" in response:
            return 1
        elif "final answer a" in response:
            return 0
        elif "final answer b" in response:
            return 1
        elif "option (a) uses the correct adjective sequence" in response:
            return 0
        elif "option (b) uses the correct adjective sequence" in response:
            return 1
        elif "the correct option is option a" in response:
            return 0
        elif "the correct option is option b" in response:
            return 1
        elif "the answer is without any modification a" in response:
            return 0
        elif "the answer is without any modification b" in response:
            return 1
        elif "the correct one is option a" in response:
            return 0
        elif "the correct one is option b" in response:
            return 1
        elif "answer is without any modification a" in response:
            return 0
        elif "answer is without any modification b" in response:
            return 1
        elif "the answer is without any modification option a" in response:
            return 0
        elif "the answer is without any modification option b" in response:
            return 1
        elif "answer is a" in response:
            return 0
        elif "answer is b" in response:
            return 1
        elif "the answer is [a]" in response:
            return 0
        elif "the answer is [b]" in response:
            return 1
        elif "option a follows the correct adjective order" in response:
            return 0
        elif "option b follows the correct adjective order" in response:
            return 1
        elif "the correct order of adjectives is a" in response:
            return 0
        elif "the correct order of adjectives is b" in response:
            return 1
        elif "option (a) is the correct answer" in response:
            return 0
        elif "option (b) is the correct answer" in response:
            return 1
        elif "the correct order is a" in response:
            return 0
        elif "the correct order is b" in response:
            return 1
        elif "the correct object order is a" in response:
            return 0
        elif "the correct object order is b" in response:
            return 1
        elif "the correct answer order is a" in response:
            return 0
        elif "the correct answer order is b" in response:
            return 1
        elif "the final answer without any modification is a" in response:
            return 0
        elif "the final answer without any modification is b" in response:
            return 1
        elif "the correct sentence is a" in response:
            return 0
        elif "the correct sentence is b" in response:
            return 1
        elif "correct adjective order option is a" in response:
            return 0
        elif "correct adjective order option is b" in response:
            return 1
        elif "correct adjective order is sentence a" in response:
            return 0
        elif "correct adjective order is sentence b" in response:
            return 1
        elif "correct objective order option is a" in response:
            return 0
        elif "correct objective order option is b" in response:
            return 1
        elif "the answer is is a" in response:
            return 0
        elif "the answer is is b" in response:
            return 1
        elif "the correct adjective order is option a" in response:
            return 0
        elif "the correct adjective order is option b" in response:
            return 1
        elif "the correct adjective order is provided in option a" in response:
            return 0
        elif "the correct adjective order is provided in option b" in response:
            return 1
        elif "option a is more accurate" in response:
            return 0
        elif "option b is more accurate" in response:
            return 1
        elif "answer is option a" in response:
            return 0
        elif "answer is option b" in response:
            return 1
        elif "option a has the adjectives in the correct order" in response:
            return 0
        elif "option b has the adjectives in the correct order" in response:
            return 1
        elif "option a is closer to being correct" in response:
            return 0
        elif "option b is closer to being correct" in response:
            return 1
        elif "the most logical order is option a" in response:
            return 0
        elif "the most logical order is option b" in response:
            return 1
        elif "in option a the adjectives are in the correct order" in response:
            return 0
        elif "in option b the adjectives are in the correct order" in response:
            return 1
        elif "sentence a is the closest to the correct adjective order" in response:
            return 0
        elif "sentence b is the closest to the correct adjective order" in response:
            return 1
        elif "option a has the correct order" in response:
            return 0
        elif "option b has the correct order" in response:
            return 1
        elif "sentence a has a more logical" in response:
            return 0
        elif "sentence b has a more logical" in response:
            return 1
        elif "the closest correct order is option a" in response:
            return 0
        elif "the closest correct order is option b" in response:
            return 1
        elif "sentence a has the correct adjective order" in response:
            return 0
        elif "sentence b has the correct adjective order" in response:
            return 1
        elif "option a is the closest to the typical order" in response:
            return 0
        elif "option b is the closest to the typical order" in response:
            return 1
        elif "the correct answer would be option a" in response:
            return 0
        elif "the correct answer would be option b" in response:
            return 1
        elif "the correct option is a" in response:
            return 0
        elif "the correct option is b" in response:
            return 1
        elif "option a has a better adjective order" in response:
            return 0
        elif "option b has a better adjective order" in response:
            return 1
        elif "option a has the correct adjective order" in response:
            return 0
        elif "option b has the correct adjective order" in response:
            return 1
        elif "option a seems to follow the typical order" in response:
            return 0
        elif "option b seems to follow the typical order" in response:
            return 1
        elif "the correct order is found in option a" in response:
            return 0
        elif "the correct order is found in option b" in response:
            return 1
        elif "the correct adjective order is in the first option" in response:
            return 0
        elif "the correct adjective order is in the second option" in response:
            return 1
        elif "the correct adverb order would be a" in response:
            return 0
        elif "the correct adverb order would be b" in response:
            return 1
        elif "the correct answer would be the a" in response:
            return 0
        elif "the correct answer would be the b" in response:
            return 1
        elif "the correct adjective order is in option a" in response:
            return 0
        elif "the correct adjective order is in option b" in response:
            return 1
        elif "the correct adjective order is in sentence a" in response:
            return 0
        elif "the correct adjective order is in sentence b" in response:
            return 1
        elif "the adjectives are in the correct order for a" in response:
            return 0
        elif "the adjectives are in the correct order for b" in response:
            return 1
        elif "the answer is [option a]" in response:
            return 0
        elif "the answer is [option b]" in response:
            return 1
        elif "option (a) has a correct adjective order" in response:
            return 0
        elif "option (b) has a correct adjective order" in response:
            return 1
        elif "the correct sentence would be a" in response:
            return 0
        elif "the correct sentence would be b" in response:
            return 1
        elif "option a follows the correct sequence of adjectives" in response:
            return 0
        elif "option b follows the correct sequence of adjectives" in response:
            return 1
        elif "option a would be closer to the correct order" in response:
            return 0
        elif "option b would be closer to the correct order" in response:
            return 1
        elif "the correct sentence with the adjective order is a" in response:
            return 0
        elif "the correct sentence with the adjective order is b" in response:
            return 1
        elif "option a is more grammatically correct" in response:
            return 0
        elif "option b is more grammatically correct" in response:
            return 1
        elif "option a would be more correct" in response:
            return 0
        elif "option b would be more correct" in response:
            return 1
        elif "option a is incorrect and option b is correct" in response:
            return 1
        elif "option a is the closest match" in response:
            return 0
        elif "option b is the closest match" in response:
            return 1
        elif "option b follows the typical order of adjectives" in response:
            return 1
        elif "the correct sentence with the adjective order is option a" in response:
            return 0
        elif "the correct sentence with the adjective order is option b" in response:
            return 1
        elif "the correct sentence is option a" in response:
            return 0
        elif "the correct sentence is option b" in response:
            return 1
        elif "the correct objective order is in option a" in response:
            return 0
        elif "the correct objective order is in option b" in response:
            return 1
        elif "the answer is option (a)" in response:
            return 0
        elif "the answer is option (b)" in response:
            return 1
        elif "the correct option would be (a)" in response:
            return 0
        elif "the correct option would be (b)" in response:
            return 1
        elif "the correct order is in option a" in response:
            return 0
        elif "the correct order is in option b" in response:
            return 1
        elif "the correct adjective order is found in option a" in response:
            return 0
        elif "the correct adjective order is found in option b" in response:
            return 1
        elif "option (a) follows the typical order of adjectives" in response:
            return 0
        elif "option (b) follows the typical order of adjectives" in response:
            return 1
        elif "option (a) has the correct adjective order" in response:
            return 0
        elif "option (b) has the correct adjective order" in response:
            return 1
        elif "option (a) follows the general adjective order" in response:
            return 0
        elif "option (b) follows the general adjective order" in response:
            return 1
        elif "option (a) would be more grammatically correct" in response:
            return 0
        elif "option (b) would be more grammatically correct" in response:
            return 1
        elif "option a is the correct" in response:
            return 0
        elif "option b is the correct" in response:
            return 1
        elif "option a has the correct word order" in response:
            return 0
        elif "option b has the correct word order" in response:
            return 1
        elif "option a follows the correct order" in response:
            return 0
        elif "option b follows the correct order" in response:
            return 1
        elif "following the typical order is the first option" in response:
            return 0
        elif "following the typical order is the second option" in response:
            return 1
        elif "option a has a slightly better chance of being correct" in response:
            return 0
        elif "option b has a slightly better chance of being correct" in response:
            return 1
        elif "the answer is sentence a" in response:
            return 0
        elif "the answer is sentence b" in response:
            return 1
        elif "the final answer is (a)" in response:
            return 0
        elif "the final answer is (b)" in response:
            return 1
        elif re.search(r"sentence a (.+?) seems to have a more logical", response):
            return 0
        elif re.search(r"sentence b (.+?) seems to have a more logical", response):
            return 1
        elif re.search(r"the correct adjective order is (.+?) option a", response):
            return 0
        elif re.search(r"the correct adjective order is (.+?) option b", response):
            return 1
        elif response.startswith("a "):
            return 0
        elif response.startswith("b "):
            return 1
        elif response.startswith("a)"):
            return 0
        elif response.startswith("b)"):
            return 1
        else:
            return random.choice([0, 1])

    def extract_answer_yn(self, response):
        if "answer is no" in response:
            return 0
        elif "answer is yes" in response:
            return 1
        elif "the answer no" in response:
            return 0
        elif "the answer yes" in response:
            return 1
        elif "final answer no" in response:
            return 0
        elif "final answer yes" in response:
            return 1
        elif "i would answer no" in response:
            return 0
        elif "i would answer yes" in response:
            return 1
        elif "the answer is without any modification no" in response:
            return 0
        elif "the answer is without any modification yes" in response:
            return 1
        elif "the answer to the question is no" in response:
            return 0
        elif "the answer to the question is yes" in response:
            return 1
        elif "the answer is false" in response:
            return 0
        elif "the answer is true" in response:
            return 1
        elif "answer false" in response:
            return 0
        elif "answer true" in response:
            return 1
        elif "the answer is $\\boxed{\\text{false}}$" in response:
            return 0
        elif "the answer is $\\boxed{\\text{true}}$" in response:
            return 1
        elif "the answer is \\boxed{no}" in response:
            return 0
        elif "the answer is \\boxed{yes}" in response:
            return 1
        elif "the answer is \\boxed{\\text{false}}" in response:
            return 0
        elif "the answer is \\boxed{\\text{true}}" in response:
            return 1
        elif "the answer is \\boxed{\\text{no}}" in response:
            return 0
        elif "the answer is \\boxed{\\text{yes}}" in response:
            return 1
        elif "i will provide the answer as no" in response:
            return 0
        elif "i will provide the answer as yes" in response:
            return 1
        elif "prefix answer no" in response:
            return 0
        elif "prefix answer yes" in response:
            return 1
        elif "conclusion no" in response:
            return 0
        elif "conclusion yes" in response:
            return 1
        else:
            return None

    def extract_answer_lies(self, response):
        tmp = self.extract_answer_yn(response)
        if tmp is not None:
            return tmp
        if "the prefix no applies to the statement" in response:
            return 0
        elif "veena s statement cannot be true" in response:
            return 0
        elif "therefore truly lorene tells the truth" in response:
            return 1
        elif "the answer is the truth" in response:
            return 1
        elif "affirmatively alejandro tells the truth" in response:
            return 1
        elif re.search(r"answer is (.+?) tells a lie", response):
            return 0
        elif re.search(r"answer is (.+?) lies", response):
            return 0
        elif re.search(r"answer is (.+?) says lie", response):
            return 0
        elif re.search(r"answer is (.+?) doesn t tell the truth", response):
            return 0
        elif re.search(r"answer is (.+?) does not tell the truth", response):
            return 0
        elif re.search(r"answer is (.+?) didn t tell the truth", response):
            return 0
        elif re.search(r"answer is (.+?) tells the truth", response):
            return 1
        elif re.search(r"answer is (.+?) does tell the truth", response):
            return 1
        elif re.search(r"answer to the question (.+?) is no", response):
            return 0
        elif re.search(r"answer to the question (.+?) is yes", response):
            return 1
        elif re.search(r"from the above steps we can conclude that (.+?) tells the truth", response):
            return 1
        elif response.endswith("does not tell the truth"):
            return 0
        elif response.endswith("cannot be telling the truth"):
            return 0
        elif response.endswith("is lying"):
            return 0
        elif response.endswith("tells the lie"):
            return 0
        elif response.endswith("is also telling the truth"):
            return 1
        elif response.endswith("must be lying"):
            return 1
        elif response.endswith("must be telling the truth"):
            return 1
        elif response.endswith("tells the truth"):
            return 1
        elif response.endswith("delfina does tell the truth"):
            return 1
        elif response.endswith("osvaldo is telling the truth"):
            return 1
        elif response.endswith("lies"):
            return 0
        elif response.startswith("no"):
            return 0
        elif response.startswith("yes"):
            return 1
        elif response.endswith("no"):
            return 0
        elif response.endswith("yes"):
            return 1
        else:
            return random.choice([0, 1])

    def extract_answer_navigate(self, response):
        tmp = self.extract_answer_yn(response)
        if tmp is not None:
            return tmp
        if "you do not return to the starting point" in response:
            return 0
        elif "you are not at the starting point" in response:
            return 0
        elif "you haven t moved back to the starting point" in response:
            return 0
        elif "you cannot return to the starting point" in response:
            return 0
        elif "you return to the starting point" in response:
            return 1
        elif "you are not facing the starting point" in response:
            return 0
        elif "you haven t returned to the starting point" in response:
            return 0
        elif "you end up back at the starting point" in response:
            return 1
        elif "you are not back at the starting point" in response:
            return 0
        elif "you will not return to the starting point" in response:
            return 0
        elif "yes following these instructions" in response:
            return 1
        elif "we end up back at the starting point" in response:
            return 1
        elif "indeed returns us to the starting point" in response:
            return 1
        elif "indeed bring us back to the starting point" in response:
            return 1
        elif "you ll end up right back where you started" in response:
            return 1
        elif "we ve now returned to the starting point" in response:
            return 1
        elif "i have returned to the starting position" in response:
            return 1
        elif "the final position is not directly at the starting point" in response:
            return 0
        elif "it appears that we did not return to the exact starting point" in response:
            return 0
        elif "does not return us to the starting point" in response:
            return 0
        elif "i will end up back where i started" in response:
            return 1
        elif "indeed return to the starting point" in response:
            return 1
        elif "i ll be back where i started" in response:
            return 1
        elif "i ll end up back at the starting point" in response:
            return 1
        elif "after following these instructions we return to the starting point" in response:
            return 1
        elif "we are back at the starting point!" in response:
            return 1
        elif "we are now back at the starting point" in response:
            return 1
        elif "you ll end up back where you started" in response:
            return 1
        elif "we have now returned to the starting point" in response:
            return 1
        elif "i ve returned to the starting point" in response:
            return 1
        elif "following these instructions will always return us to the starting point" in response:
            return 1
        elif "indeed return you to the starting point" in response:
            return 1
        elif "the answer is the starting point" in response:
            return 1
        elif "following these instructions doesn t return us to the starting point" in response:
            return 0
        elif "following these directions does not lead you back to your original starting point" in response:
            return 0
        elif response.startswith("no"):
            return 0
        elif response.startswith("yes"):
            return 1
        elif response.endswith("no"):
            return 0
        elif response.endswith("yes"):
            return 1
        else:
            return random.choice([0, 1])

    def extract_answer_sports(self, response):
        tmp = self.extract_answer_yn(response)
        if tmp is not None:
            return tmp
        if "final answer the sentence is plausible" in response:
            return 1
        elif "the answer is directly yes" in response:
            return 1
        elif "no the sentence is not possible" in response:
            return 0
        elif "the answer is it is not possible" in response:
            return 0
        elif "the answer is the sentence is plausible" in response:
            return 1
        elif "the sentence is not particularly plausible" in response:
            return 0
        elif "the sentence is not plausible" in response:
            return 0
        elif "yes the sentence is structurally plausible" in response:
            return 1
        elif "is the sentence plausible? yes it is" in response:
            return 1
        elif "i would say that it is not a plausible sentence." in response:
            return 0
        elif "no the original sentence is not plausible" in response:
            return 0
        elif "it s not a plausible" in response:
            return 0
        elif "to answer your original question no" in response:
            return 0
        elif "making it plausible" in response:
            return 1
        elif "making it implausible" in response:
            return 0
        elif "it is indeed a plausible sentence" in response:
            return 1
        elif "considering these points the sentence is plausible" in response:
            return 1
        elif "i would say the sentence is plausible" in response:
            return 1
        elif "i would say that the sentence is plausible" in response:
            return 1
        elif "i d say it s not entirely plausible" in response:
            return 0
        elif "therefore not plausible" in response:
            return 0
        elif "making it an implausible statement" in response:
            return 0
        elif "the following sentence is not plausible" in response:
            return 0
        elif "considering these points the sentence is unlikely to be true" in response:
            return 0
        elif "considering these points the sentence is not plausible" in response:
            return 0
        elif "yes the sentence is plausible" in response:
            return 1
        elif "based on this analysis the sentence is plausible" in response:
            return 1
        elif "considering these points the sentence seems plausible" in response:
            return 1
        elif "given the context the sentence is plausible" in response:
            return 1
        elif "considering these factors the sentence is plausible" in response:
            return 1
        elif "considering these points the sentence is unlikely to be plausible" in response:
            return 0
        elif "given the context of sports particularly basketball this sentence is plausible" in response:
            return 1
        elif "considering these points the sentence is likely true" in response:
            return 1
        elif "considering these elements the sentence is plausible" in response:
            return 1
        elif "i would say it s unlikely" in response:
            return 0
        elif "it seems unlikely" in response:
            return 0
        elif "given this analysis the sentence is plausible" in response:
            return 1
        elif "considering these points the sentence is the plausible" in response:
            return 1
        elif "the sentence is not entirely possible" in response:
            return 0
        elif "the sentence is not entirely accurate" in response:
            return 0
        elif "the sentence is not entirely plausible" in response:
            return 0
        elif "the answer is plausible" in response:
            return 1
        elif "modification of answer not possible" in response:
            return 0
        elif "the answer is without any modification no" in response:
            return 0
        elif "i would say that the sentence is generally plausible" in response:
            return 1
        elif "given these points the sentence is plausible" in response:
            return 1
        elif "yes the sentence is plausible" in response:
            return 1
        elif "considering these points the sentence seems to be a plausible" in response:
            return 1
        elif "the sentence appears to be a plausible" in response:
            return 1
        elif response == "not plausible":
            return 0
        elif re.search(
            r"considering these points the sentence (.+?) is grammatically correct and makes sense", response
        ):
            return 1
        elif re.search(r"considering these points the sentence (.+?) is plausible", response):
            return 1
        elif re.search(r"considering these points the sentence (.+?) is unlikely", response):
            return 0
        elif re.search(r"based on this analysis the sentence (.+?) is plausible", response):
            return 1
        elif re.search(r"considering these points the sentence (.+?) seems to be a possible", response):
            return 1
        elif re.search(r"based on these steps the sentence (.+?) is plausible", response):
            return 1
        elif re.search(r"based on this analysis the sentence (.+?) seems plausible", response):
            return 1
        elif re.search(r"considering these points the sentence (.+?) appears to be plausible", response):
            return 1
        elif re.search(r"considering these points the sentence (.+?) seems plausible", response):
            return 1
        elif re.search(r"the sentence (.+?) is not very plausible", response):
            return 0
        elif re.search(r"considering these points (.+?) is a plausible sentence", response):
            return 1
        elif re.search(r"given this information the sentence (.+?) is plausible", response):
            return 1
        elif re.search(r"given the context the sentence (.+?) is plausible", response):
            return 1
        elif re.search(r"the sentence (.+?) is flexible and realistic", response):
            return 1
        elif re.search("considering these points the sentence (.+?) is not a plausible statement", response):
            return 0
        elif re.search(r"the sentence (.+?) is not plausible", response):
            return 1
        elif response.startswith("no"):
            return 0
        elif response.startswith("yes"):
            return 1
        elif response.startswith("the following sentence is plausible"):
            return 1
        elif response.endswith("no"):
            return 0
        elif response.endswith("yes"):
            return 1
        else:
            return random.choice([0, 1])
