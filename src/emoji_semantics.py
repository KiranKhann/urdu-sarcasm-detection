# -*- coding: utf-8 -*-
"""
Emoji semantics for Urdu sarcasm detection.

Social-media Urdu relies heavily on emojis (83% of posts in this dataset contain
at least one).  An emoji is not noise: it carries (a) a literal meaning,
(b) a sentiment polarity and (c) a pragmatic function (laughing *at* vs. laughing
*with*, a smirk, an eye-roll).  Sarcasm is often signalled by a *mismatch*
between the words and the emoji (praise + 😂, complaint + 😍, "great job" + 🙄).

This module turns every emoji into something a model can reason about:

    describe(text)         -> list of EmojiInfo (meaning in English + Urdu, polarity, category)
    explain(text)          -> human readable explanation string
    emoji_to_tokens(text)  -> text with emojis rewritten as word-like tokens
    emoji_features(text)   -> dense numeric features (counts, polarity, contrast, position)
    EmojiPrior             -> data-driven sarcasm log-odds per emoji, fitted on TRAIN only
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import emoji as _emoji

# --------------------------------------------------------------------------- #
# 1. Curated lexicon                                                          #
# --------------------------------------------------------------------------- #
# pol : +1 positive, 0 neutral/ambiguous, -1 negative
# cat : coarse pragmatic class used for grouped tokens / counts
# cue : True when the emoji frequently flags irony/mockery in Pakistani Twitter
#       usage (laughing at someone, smirk, eye-roll, tongue-out).
# ur  : short Urdu gloss so the *meaning* of the emoji is available as text.

LEXICON: Dict[str, dict] = {
    # ---- laughter -----------------------------------------------------------
    "😂": dict(en="face with tears of joy",         ur="ہنسی کے آنسو",            pol=+1, cat="LAUGH", cue=True),
    "🤣": dict(en="rolling on the floor laughing",  ur="لوٹ پوٹ ہنسی",            pol=+1, cat="LAUGH", cue=True),
    "😆": dict(en="grinning squinting face",        ur="زور کی ہنسی",             pol=+1, cat="LAUGH", cue=True),
    "😅": dict(en="grinning face with sweat",       ur="شرمندہ ہنسی",             pol=0,  cat="LAUGH", cue=True),
    "😁": dict(en="beaming face with smiling eyes", ur="دانت نکال کر ہنسی",        pol=+1, cat="LAUGH", cue=True),
    "😄": dict(en="grinning face with smiling eyes",ur="کھلی ہنسی",               pol=+1, cat="LAUGH", cue=True),
    "😃": dict(en="grinning face with big eyes",    ur="کھلی ہنسی",               pol=+1, cat="LAUGH", cue=True),
    "😀": dict(en="grinning face",                  ur="ہنسی",                    pol=+1, cat="LAUGH", cue=True),
    "🤭": dict(en="face with hand over mouth",      ur="دبی ہنسی",                pol=0,  cat="LAUGH", cue=True),
    "😹": dict(en="cat with tears of joy",          ur="ہنستی بلی",               pol=+1, cat="LAUGH", cue=True),
    "🫠": dict(en="melting face",                   ur="پگھلتا چہرہ",             pol=0,  cat="LAUGH", cue=True),
    # ---- tongue-out / teasing ----------------------------------------------
    "😜": dict(en="winking face with tongue",       ur="شرارتی چہرہ",             pol=0,  cat="TEASE", cue=True),
    "😝": dict(en="squinting face with tongue",     ur="زبان نکالنا",             pol=0,  cat="TEASE", cue=True),
    "😛": dict(en="face with tongue",               ur="زبان نکالنا",             pol=0,  cat="TEASE", cue=True),
    "😋": dict(en="face savoring food",             ur="مزے لینا",                pol=+1, cat="TEASE", cue=True),
    "😉": dict(en="winking face",                   ur="آنکھ مارنا",              pol=0,  cat="TEASE", cue=True),
    "😎": dict(en="smiling face with sunglasses",   ur="کول / بے پروا",            pol=+1, cat="TEASE", cue=True),
    "🤪": dict(en="zany face",                      ur="پاگل پن",                 pol=0,  cat="TEASE", cue=True),
    # ---- smirk / irony markers ---------------------------------------------
    "😏": dict(en="smirking face",                  ur="طنزیہ مسکراہٹ",            pol=0,  cat="SMIRK", cue=True),
    "🙄": dict(en="face with rolling eyes",         ur="آنکھیں گھمانا / بیزاری",   pol=-1, cat="SMIRK", cue=True),
    "😒": dict(en="unamused face",                  ur="ناخوش / بیزار",            pol=-1, cat="SMIRK", cue=True),
    "🙂": dict(en="slightly smiling face",          ur="ہلکی (اکثر طنزیہ) مسکراہٹ", pol=0,  cat="SMIRK", cue=True),
    "🙃": dict(en="upside-down face",               ur="الٹا چہرہ / طنز",          pol=0,  cat="SMIRK", cue=True),
    "😬": dict(en="grimacing face",                 ur="دانت پیسنا",              pol=-1, cat="SMIRK", cue=True),
    "😑": dict(en="expressionless face",            ur="بے تاثر",                 pol=-1, cat="SMIRK", cue=True),
    "😐": dict(en="neutral face",                   ur="بے تاثر چہرہ",             pol=0,  cat="SMIRK", cue=True),
    "🤡": dict(en="clown face",                     ur="مسخرہ",                   pol=-1, cat="SMIRK", cue=True),
    "🤷": dict(en="person shrugging",               ur="کندھے اچکانا",             pol=0,  cat="SMIRK", cue=True),
    "🤦": dict(en="person facepalming",             ur="ہاتھ ماتھے پر / افسوس",     pol=-1, cat="SMIRK", cue=True),
    "😇": dict(en="smiling face with halo",         ur="معصوم بننا",              pol=0,  cat="SMIRK", cue=True),
    "🤐": dict(en="zipper-mouth face",              ur="منہ بند",                 pol=0,  cat="SMIRK", cue=True),
    "😶": dict(en="face without mouth",             ur="خاموش",                   pol=0,  cat="SMIRK", cue=True),
    # ---- thinking / doubt ---------------------------------------------------
    "🤔": dict(en="thinking face",                  ur="سوچنا / شک",               pol=0,  cat="THINK", cue=True),
    "🧐": dict(en="face with monocle",              ur="غور سے دیکھنا",            pol=0,  cat="THINK", cue=True),
    "🤨": dict(en="face with raised eyebrow",       ur="شک بھری نظر",              pol=0,  cat="THINK", cue=True),
    "😕": dict(en="confused face",                  ur="الجھن",                   pol=-1, cat="THINK", cue=False),
    # ---- love / affection ---------------------------------------------------
    "❤":  dict(en="red heart",                      ur="دل / محبت",                pol=+1, cat="LOVE", cue=False),
    "♥":  dict(en="heart suit",                     ur="دل / محبت",                pol=+1, cat="LOVE", cue=False),
    "💕": dict(en="two hearts",                     ur="محبت",                    pol=+1, cat="LOVE", cue=False),
    "💞": dict(en="revolving hearts",               ur="محبت",                    pol=+1, cat="LOVE", cue=False),
    "💖": dict(en="sparkling heart",                ur="چمکتا دل",                pol=+1, cat="LOVE", cue=False),
    "💗": dict(en="growing heart",                  ur="دل",                      pol=+1, cat="LOVE", cue=False),
    "💓": dict(en="beating heart",                  ur="دھڑکتا دل",               pol=+1, cat="LOVE", cue=False),
    "💘": dict(en="heart with arrow",               ur="عشق",                     pol=+1, cat="LOVE", cue=False),
    "💙": dict(en="blue heart",                     ur="دل",                      pol=+1, cat="LOVE", cue=False),
    "💚": dict(en="green heart",                    ur="دل",                      pol=+1, cat="LOVE", cue=False),
    "💛": dict(en="yellow heart",                   ur="دل",                      pol=+1, cat="LOVE", cue=False),
    "💜": dict(en="purple heart",                   ur="دل",                      pol=+1, cat="LOVE", cue=False),
    "🖤": dict(en="black heart",                    ur="کالا دل",                 pol=0,  cat="LOVE", cue=False),
    "🤍": dict(en="white heart",                    ur="دل",                      pol=+1, cat="LOVE", cue=False),
    "😍": dict(en="smiling face with heart-eyes",   ur="پیار بھری نظر",            pol=+1, cat="LOVE", cue=False),
    "🥰": dict(en="smiling face with hearts",       ur="پیار",                    pol=+1, cat="LOVE", cue=False),
    "😘": dict(en="face blowing a kiss",            ur="بوسہ",                    pol=+1, cat="LOVE", cue=False),
    "😻": dict(en="smiling cat with heart-eyes",    ur="پیار بھری بلی",            pol=+1, cat="LOVE", cue=False),
    "😊": dict(en="smiling face with smiling eyes", ur="مسکراہٹ",                 pol=+1, cat="LOVE", cue=False),
    "☺":  dict(en="smiling face",                   ur="مسکراہٹ",                 pol=+1, cat="LOVE", cue=False),
    "😌": dict(en="relieved face",                  ur="اطمینان",                 pol=+1, cat="LOVE", cue=False),
    # ---- sadness / heartbreak (poetry / shayari register) ------------------
    "💔": dict(en="broken heart",                   ur="ٹوٹا دل",                 pol=-1, cat="SAD", cue=False),
    "😭": dict(en="loudly crying face",             ur="زور سے رونا",              pol=-1, cat="SAD", cue=False),
    "😢": dict(en="crying face",                    ur="رونا",                    pol=-1, cat="SAD", cue=False),
    "😥": dict(en="sad but relieved face",          ur="اداس",                    pol=-1, cat="SAD", cue=False),
    "😔": dict(en="pensive face",                   ur="اداس / فکرمند",            pol=-1, cat="SAD", cue=False),
    "😞": dict(en="disappointed face",              ur="مایوس",                   pol=-1, cat="SAD", cue=False),
    "😣": dict(en="persevering face",               ur="پریشان",                  pol=-1, cat="SAD", cue=False),
    "😩": dict(en="weary face",                     ur="تھکا ہارا",                pol=-1, cat="SAD", cue=False),
    "😫": dict(en="tired face",                     ur="تھکا",                    pol=-1, cat="SAD", cue=False),
    "🥺": dict(en="pleading face",                  ur="التجا",                   pol=-1, cat="SAD", cue=False),
    "😓": dict(en="downcast face with sweat",       ur="اداس / شرمندہ",            pol=-1, cat="SAD", cue=False),
    "🥀": dict(en="wilted flower",                  ur="مرجھایا پھول / اداسی",      pol=-1, cat="SAD", cue=False),
    "🍂": dict(en="fallen leaf",                    ur="گرا پتا / خزاں",           pol=-1, cat="SAD", cue=False),
    "🍁": dict(en="maple leaf",                     ur="خزاں کا پتا",              pol=0,  cat="SAD", cue=False),
    # ---- anger / disgust ----------------------------------------------------
    "😡": dict(en="enraged face",                   ur="غصہ",                     pol=-1, cat="ANGER", cue=False),
    "😠": dict(en="angry face",                     ur="غصہ",                     pol=-1, cat="ANGER", cue=False),
    "🤬": dict(en="face with symbols on mouth",     ur="گالی",                    pol=-1, cat="ANGER", cue=False),
    "😤": dict(en="face with steam from nose",      ur="ناراض",                   pol=-1, cat="ANGER", cue=False),
    "🤮": dict(en="face vomiting",                  ur="قے / نفرت",                pol=-1, cat="ANGER", cue=True),
    "🤢": dict(en="nauseated face",                 ur="متلی",                    pol=-1, cat="ANGER", cue=True),
    "💩": dict(en="pile of poo",                    ur="گندگی",                   pol=-1, cat="ANGER", cue=True),
    "👎": dict(en="thumbs down",                    ur="ناپسند",                  pol=-1, cat="ANGER", cue=False),
    "😈": dict(en="smiling face with horns",        ur="شیطانی مسکراہٹ",            pol=-1, cat="ANGER", cue=True),
    # ---- prayer / religion / gratitude -------------------------------------
    "🙏": dict(en="folded hands",                   ur="دعا / شکریہ",              pol=+1, cat="PRAYER", cue=False),
    "🤲": dict(en="palms up together",              ur="دعا",                     pol=+1, cat="PRAYER", cue=False),
    "☪":  dict(en="star and crescent",              ur="چاند تارہ",                pol=+1, cat="PRAYER", cue=False),
    "🕋": dict(en="kaaba",                          ur="کعبہ",                    pol=+1, cat="PRAYER", cue=False),
    "🕌": dict(en="mosque",                         ur="مسجد",                    pol=+1, cat="PRAYER", cue=False),
    "📿": dict(en="prayer beads",                   ur="تسبیح",                   pol=+1, cat="PRAYER", cue=False),
    # ---- praise / approval --------------------------------------------------
    "👍": dict(en="thumbs up",                      ur="پسند / ٹھیک",              pol=+1, cat="PRAISE", cue=False),
    "👌": dict(en="OK hand",                        ur="ٹھیک / بہترین",            pol=+1, cat="PRAISE", cue=False),
    "👏": dict(en="clapping hands",                 ur="تالی / شاباش",             pol=+1, cat="PRAISE", cue=True),
    "💯": dict(en="hundred points",                 ur="سو فیصد / بالکل",          pol=+1, cat="PRAISE", cue=False),
    "🔥": dict(en="fire",                           ur="آگ / زبردست",              pol=+1, cat="PRAISE", cue=False),
    "✌":  dict(en="victory hand",                   ur="فتح",                     pol=+1, cat="PRAISE", cue=False),
    "💪": dict(en="flexed biceps",                  ur="طاقت",                    pol=+1, cat="PRAISE", cue=False),
    "🤝": dict(en="handshake",                      ur="ہاتھ ملانا",               pol=+1, cat="PRAISE", cue=False),
    "🫡": dict(en="saluting face",                  ur="سلیوٹ",                   pol=+1, cat="PRAISE", cue=True),
    "🎉": dict(en="party popper",                   ur="جشن / مبارک",              pol=+1, cat="PRAISE", cue=False),
    "🥳": dict(en="partying face",                  ur="جشن",                     pol=+1, cat="PRAISE", cue=False),
    "👑": dict(en="crown",                          ur="تاج",                     pol=+1, cat="PRAISE", cue=False),
    "⭐": dict(en="star",                           ur="ستارہ",                   pol=+1, cat="PRAISE", cue=False),
    "✨": dict(en="sparkles",                       ur="چمک",                     pol=+1, cat="PRAISE", cue=False),
    # ---- flowers / nature (poetry, greetings) ------------------------------
    "🌹": dict(en="rose",                           ur="گلاب",                    pol=+1, cat="FLOWER", cue=False),
    "🌷": dict(en="tulip",                          ur="پھول",                    pol=+1, cat="FLOWER", cue=False),
    "🌺": dict(en="hibiscus",                       ur="پھول",                    pol=+1, cat="FLOWER", cue=False),
    "🌸": dict(en="cherry blossom",                 ur="پھول",                    pol=+1, cat="FLOWER", cue=False),
    "🌻": dict(en="sunflower",                      ur="سورج مکھی",               pol=+1, cat="FLOWER", cue=False),
    "🌼": dict(en="blossom",                        ur="پھول",                    pol=+1, cat="FLOWER", cue=False),
    "💐": dict(en="bouquet",                        ur="گلدستہ",                  pol=+1, cat="FLOWER", cue=False),
    "🍃": dict(en="leaf fluttering in wind",        ur="پتا",                     pol=+1, cat="FLOWER", cue=False),
    "🌙": dict(en="crescent moon",                  ur="چاند",                    pol=+1, cat="FLOWER", cue=False),
    # ---- shy / embarrassed / surprise --------------------------------------
    "🙈": dict(en="see-no-evil monkey",             ur="شرم / آنکھیں بند",          pol=0,  cat="SHY", cue=True),
    "🙊": dict(en="speak-no-evil monkey",           ur="چپ / منہ بند",             pol=0,  cat="SHY", cue=True),
    "🙉": dict(en="hear-no-evil monkey",            ur="کان بند",                 pol=0,  cat="SHY", cue=True),
    "😳": dict(en="flushed face",                   ur="شرمندہ / حیران",           pol=0,  cat="SHY", cue=False),
    "😮": dict(en="face with open mouth",           ur="حیران",                   pol=0,  cat="SHY", cue=False),
    "😱": dict(en="face screaming in fear",         ur="خوف / حیرت",               pol=-1, cat="SHY", cue=True),
    # ---- pointing / gestures ------------------------------------------------
    "👇": dict(en="backhand index pointing down",   ur="نیچے اشارہ",               pol=0,  cat="GESTURE", cue=False),
    "👆": dict(en="backhand index pointing up",     ur="اوپر اشارہ",               pol=0,  cat="GESTURE", cue=False),
    "👉": dict(en="backhand index pointing right",  ur="اشارہ",                   pol=0,  cat="GESTURE", cue=False),
    "👈": dict(en="backhand index pointing left",   ur="اشارہ",                   pol=0,  cat="GESTURE", cue=False),
    "👊": dict(en="oncoming fist",                  ur="مکا",                     pol=0,  cat="GESTURE", cue=False),
    "✋": dict(en="raised hand",                    ur="ہاتھ",                    pol=0,  cat="GESTURE", cue=False),
    "🙋": dict(en="person raising hand",            ur="ہاتھ اٹھانا",              pol=0,  cat="GESTURE", cue=False),
    "🏃": dict(en="person running",                 ur="بھاگنا",                  pol=0,  cat="GESTURE", cue=True),
    # ---- animals used as insults in Urdu/Punjabi banter --------------------
    "🐶": dict(en="dog face",                       ur="کتا",                     pol=-1, cat="INSULT", cue=True),
    "🐕": dict(en="dog",                            ur="کتا",                     pol=-1, cat="INSULT", cue=True),
    "🐐": dict(en="goat",                           ur="بکرا",                    pol=-1, cat="INSULT", cue=True),
    "🐍": dict(en="snake",                          ur="سانپ",                    pol=-1, cat="INSULT", cue=True),
    "🐒": dict(en="monkey",                         ur="بندر",                    pol=-1, cat="INSULT", cue=True),
    "🐂": dict(en="ox",                             ur="بیل",                     pol=-1, cat="INSULT", cue=True),
    "🐃": dict(en="water buffalo",                  ur="بھینس",                   pol=-1, cat="INSULT", cue=True),
    "🐉": dict(en="dragon",                         ur="اژدہا",                   pol=0,  cat="INSULT", cue=True),
    "🤥": dict(en="lying face",                     ur="جھوٹا",                   pol=-1, cat="INSULT", cue=True),
    # ---- additional emojis frequent in this corpus -------------------------
    "❣":  dict(en="heart exclamation",              ur="دل / محبت",                pol=+1, cat="LOVE", cue=False),
    "💝": dict(en="heart with ribbon",              ur="تحفہ دل",                 pol=+1, cat="LOVE", cue=False),
    "💌": dict(en="love letter",                    ur="محبت نامہ",               pol=+1, cat="LOVE", cue=False),
    "🤗": dict(en="smiling face with open hands",   ur="گلے لگانا",               pol=+1, cat="LOVE", cue=False),
    "😚": dict(en="kissing face with closed eyes",  ur="بوسہ",                    pol=+1, cat="LOVE", cue=False),
    "😗": dict(en="kissing face",                   ur="بوسہ",                    pol=+1, cat="LOVE", cue=True),
    "😷": dict(en="face with medical mask",         ur="ماسک والا چہرہ",           pol=0,  cat="SMIRK", cue=True),
    "🌚": dict(en="new moon face",                  ur="کالا چاند / شرارت",        pol=0,  cat="SMIRK", cue=True),
    "🤓": dict(en="nerd face",                      ur="پڑھاکو / عینک والا",       pol=0,  cat="TEASE", cue=True),
    "🤫": dict(en="shushing face",                  ur="چپ رہو",                  pol=0,  cat="SMIRK", cue=True),
    "🥸": dict(en="disguised face",                 ur="بھیس بدلا چہرہ",           pol=0,  cat="TEASE", cue=True),
    "🍿": dict(en="popcorn",                        ur="پاپ کارن / تماشا",         pol=0,  cat="TEASE", cue=True),
    "👅": dict(en="tongue",                         ur="زبان",                    pol=0,  cat="TEASE", cue=True),
    "☹":  dict(en="frowning face",                  ur="اداس چہرہ",               pol=-1, cat="SAD", cue=False),
    "🙁": dict(en="slightly frowning face",         ur="اداس",                    pol=-1, cat="SAD", cue=False),
    "😟": dict(en="worried face",                   ur="فکرمند",                  pol=-1, cat="SAD", cue=False),
    "😰": dict(en="anxious face with sweat",        ur="پریشان",                  pol=-1, cat="SAD", cue=False),
    "😪": dict(en="sleepy face",                    ur="نیند / تھکن",              pol=-1, cat="SAD", cue=False),
    "😲": dict(en="astonished face",                ur="حیران",                   pol=0,  cat="SHY", cue=False),
    "😯": dict(en="hushed face",                    ur="حیران",                   pol=0,  cat="SHY", cue=False),
    "🙌": dict(en="raising hands",                  ur="ہاتھ اٹھا کر خوشی",         pol=+1, cat="PRAISE", cue=False),
    "✊": dict(en="raised fist",                    ur="مکا / جدوجہد",             pol=0,  cat="GESTURE", cue=False),
    "☝":  dict(en="index pointing up",              ur="اوپر اشارہ",               pol=0,  cat="GESTURE", cue=False),
    "🖐": dict(en="hand with fingers splayed",      ur="ہاتھ",                    pol=0,  cat="GESTURE", cue=False),
    "🙆": dict(en="person gesturing OK",            ur="ٹھیک ہے",                 pol=+1, cat="GESTURE", cue=False),
    "✍":  dict(en="writing hand",                   ur="لکھنا",                   pol=0,  cat="GESTURE", cue=False),
    "🚶": dict(en="person walking",                 ur="چلنا",                    pol=0,  cat="GESTURE", cue=False),
    "💃": dict(en="woman dancing",                  ur="ناچنا",                   pol=+1, cat="PRAISE", cue=True),
    "🌴": dict(en="palm tree",                      ur="کھجور کا درخت",            pol=+1, cat="FLOWER", cue=False),
    "🌿": dict(en="herb",                           ur="سبز پتے",                 pol=+1, cat="FLOWER", cue=False),
    "🍀": dict(en="four leaf clover",               ur="خوش قسمتی",               pol=+1, cat="FLOWER", cue=False),
    "🌾": dict(en="sheaf of rice",                  ur="فصل",                     pol=+1, cat="FLOWER", cue=False),
    "❄":  dict(en="snowflake",                      ur="برف",                     pol=0,  cat="FLOWER", cue=False),
    "💫": dict(en="dizzy",                          ur="چکر / ستارے",              pol=+1, cat="PRAISE", cue=False),
    "🕊": dict(en="dove",                           ur="امن کا کبوتر",             pol=+1, cat="PRAYER", cue=False),
    "✅": dict(en="check mark button",              ur="درست / ٹک",                pol=+1, cat="PRAISE", cue=False),
    "🔄": dict(en="counterclockwise arrows button", ur="ری ٹویٹ / دہرانا",         pol=0,  cat="OTHER", cue=False),
    "🔁": dict(en="repeat button",                  ur="دہرانا",                  pol=0,  cat="OTHER", cue=False),
    "🔅": dict(en="dim button",                     ur="علامت",                   pol=0,  cat="OTHER", cue=False),
    "▪":  dict(en="black small square",             ur="نقطہ",                    pol=0,  cat="OTHER", cue=False),
    "🐅": dict(en="tiger",                          ur="شیر / ٹائیگر",             pol=0,  cat="INSULT", cue=True),
    "👺": dict(en="goblin",                         ur="بھوت / بدروح",             pol=-1, cat="INSULT", cue=True),
    "👻": dict(en="ghost",                          ur="بھوت",                    pol=0,  cat="INSULT", cue=True),
    "💥": dict(en="collision",                      ur="دھماکا",                  pol=0,  cat="OTHER", cue=False),
    "📚": dict(en="books",                          ur="کتابیں",                  pol=0,  cat="OTHER", cue=False),
    "🍕": dict(en="pizza",                          ur="پیزا",                    pol=0,  cat="OTHER", cue=True),
    "🍼": dict(en="baby bottle",                    ur="دودھ کی بوتل / بچہ",        pol=0,  cat="INSULT", cue=True),
    # ---- misc objects -------------------------------------------------------
    "💦": dict(en="sweat droplets",                 ur="پانی کے قطرے",             pol=0,  cat="OTHER", cue=False),
    "🚛": dict(en="articulated lorry",              ur="ٹرک",                     pol=0,  cat="OTHER", cue=False),
    "🎂": dict(en="birthday cake",                  ur="سالگرہ کیک",               pol=+1, cat="OTHER", cue=False),
    "☕": dict(en="hot beverage",                   ur="چائے",                    pol=+1, cat="OTHER", cue=False),
    "📢": dict(en="loudspeaker",                    ur="اعلان",                   pol=0,  cat="OTHER", cue=False),
    "🇵🇰": dict(en="flag Pakistan",                 ur="پاکستان کا جھنڈا",          pol=+1, cat="OTHER", cue=False),
}

CATEGORIES = ["LAUGH", "TEASE", "SMIRK", "THINK", "LOVE", "SAD", "ANGER",
              "PRAYER", "PRAISE", "FLOWER", "SHY", "GESTURE", "INSULT", "OTHER"]

# variation selectors, skin-tone modifiers, gender ZWJ parts -> stripped so that
# ❤️ == ❤, ✌🏻 == ✌, 🤦‍♂️ == 🤦 (they are the same cue for our purposes)
_MODIFIERS = re.compile("[\U0001F3FB-\U0001F3FF️︎‍♀♂]")


def canonical(e: str) -> str:
    """Strip skin tone / variation selectors / gender suffix from an emoji."""
    return _MODIFIERS.sub("", e) or e


@dataclass
class EmojiInfo:
    emoji: str
    en: str
    ur: str
    polarity: int
    category: str
    cue: bool
    position: str  # "start" | "end" | "middle"
    start: int
    end: int


def meaning(e: str, lang: str = "both") -> str:
    """Meaning of one emoji: lang = 'en' | 'ur' | 'both'."""
    info = _lookup(e)
    if lang == "en":
        return info["en"]
    if lang == "ur":
        return info["ur"]
    return f"{info['en']} / {info['ur']}"


def _lookup(e: str) -> dict:
    c = canonical(e)
    if c in LEXICON:
        return LEXICON[c]
    # unknown emoji -> fall back to the Unicode CLDR name, neutral polarity
    name = _emoji.demojize(c).strip(":").replace("_", " ")
    return dict(en=name, ur=name, pol=0, cat="OTHER", cue=False)


def extract(text: str) -> List[dict]:
    """Return the raw emoji spans in a text (uses the `emoji` package)."""
    return _emoji.emoji_list(text or "")


def describe(text: str) -> List[EmojiInfo]:
    """Explain every emoji in `text`: meaning (EN/UR), polarity, category, position."""
    text = text or ""
    spans = extract(text)
    out = []
    for s in spans:
        info = _lookup(s["emoji"])
        before = _emoji.replace_emoji(text[: s["match_start"]], "").strip()
        after = _emoji.replace_emoji(text[s["match_end"]:], "").strip()
        if before == "":
            pos = "start"
        elif after == "":
            pos = "end"
        else:
            pos = "middle"
        out.append(EmojiInfo(canonical(s["emoji"]), info["en"], info["ur"], info["pol"],
                             info["cat"], info["cue"], pos, s["match_start"], s["match_end"]))
    return out


def explain(text: str) -> str:
    """Human-readable explanation used in the notebook / error analysis."""
    parts = []
    for e in describe(text):
        sign = {1: "+", 0: "0", -1: "-"}[e.polarity]
        cue = ", irony-cue" if e.cue else ""
        parts.append(f"{e.emoji} = {e.en} / {e.ur} [{e.category}, pol {sign}{cue}, {e.position}]")
    return "; ".join(parts) if parts else "(no emoji)"


# --------------------------------------------------------------------------- #
# 2. Text rewriting                                                           #
# --------------------------------------------------------------------------- #
def _en_token(info: dict) -> str:
    return "EMO_" + re.sub(r"[^a-z0-9]+", "_", info["en"].lower()).strip("_")


def emoji_to_tokens(text: str, mode: str = "demojize") -> str:
    """
    Rewrite emojis so a bag-of-words / char n-gram model can use them.

    mode = "remove"    : delete emojis (ablation: "emojis are noise")
           "keep"      : leave the raw symbol (works for char n-grams, poor for words)
           "demojize"  : 😂 -> EMO_face_with_tears_of_joy              (meaning)
           "category"  : 😂 -> EMOCAT_LAUGH                              (pragmatic class)
           "both"      : 😂 -> EMO_face_with_tears_of_joy EMOCAT_LAUGH EMOPOL_POS EMOCUE
           "urdu"      : 😂 -> ہنسی کے آنسو  (Urdu gloss: lets a text model read the
                          emoji as words - useful for transformers whose vocabulary
                          splits emojis into meaningless byte pieces)
           "augment"   : 😂 -> 😂 ہنسی کے آنسو   (symbol AND its Urdu meaning: the model
                          sees the emoji as used, plus what it means in the language
                          of the post - this is the default for the transformer)
    Repeated identical emojis (😂😂😂) are collapsed to the token plus an
    EMO_REPEAT marker: *intensity* is a sarcasm cue but exact counts are sparse.
    """
    if not text:
        return ""
    if mode == "keep":
        return text
    if mode == "remove":
        return _emoji.replace_emoji(text, replace=" ")

    spans = extract(text)
    if not spans:
        return text
    out, last, prev_tok = [], 0, None
    for s in spans:
        out.append(text[last: s["match_start"]])
        info = _lookup(s["emoji"])
        if mode == "demojize":
            tok = " " + _en_token(info) + " "
        elif mode == "category":
            tok = f" EMOCAT_{info['cat']} "
        elif mode == "urdu":
            tok = " " + info["ur"].split("/")[0].strip() + " "
        elif mode == "augment":
            tok = " " + canonical(s["emoji"]) + " " + info["ur"].split("/")[0].strip() + " "
        elif mode == "both":
            pol = {1: "POS", 0: "NEU", -1: "NEG"}[info["pol"]]
            tok = " " + _en_token(info) + f" EMOCAT_{info['cat']} EMOPOL_{pol}" + (" EMOCUE" if info["cue"] else "") + " "
        else:
            raise ValueError(f"unknown emoji mode: {mode}")
        out.append(" EMO_REPEAT " if tok == prev_tok else tok)
        prev_tok = tok
        last = s["match_end"]
    out.append(text[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


# --------------------------------------------------------------------------- #
# 3. Dense features                                                           #
# --------------------------------------------------------------------------- #
EMOJI_FEATURE_NAMES = (
    ["emo_n", "emo_unique", "emo_pos", "emo_neg", "emo_neu", "emo_cue", "emo_mean_pol",
     "emo_mixed_pol", "emo_max_run", "emo_at_start", "emo_at_end", "emo_in_middle",
     "emo_only_laugh"] + [f"emo_cat_{c}" for c in CATEGORIES]
)


def emoji_features(text: str) -> List[float]:
    infos = describe(text)
    n = len(infos)
    if n == 0:
        return [0.0] * len(EMOJI_FEATURE_NAMES)
    pols = [i.polarity for i in infos]
    cats = Counter(i.category for i in infos)
    max_run, run = 1, 1                      # longest run of the same emoji (😂😂😂😂 -> 4)
    for a, b in zip(infos, infos[1:]):
        run = run + 1 if a.emoji == b.emoji else 1
        max_run = max(max_run, run)
    feats = [
        n, len({i.emoji for i in infos}),
        sum(p > 0 for p in pols), sum(p < 0 for p in pols), sum(p == 0 for p in pols),
        sum(i.cue for i in infos), sum(pols) / n,
        float(any(p > 0 for p in pols) and any(p < 0 for p in pols)),
        max_run,
        float(any(i.position == "start" for i in infos)),
        float(any(i.position == "end" for i in infos)),
        float(any(i.position == "middle" for i in infos)),
        float(all(i.category == "LAUGH" for i in infos)),
    ] + [cats.get(c, 0) for c in CATEGORIES]
    return [float(x) for x in feats]


# --------------------------------------------------------------------------- #
# 4. Data-driven emoji prior (fit on train split only -> no leakage)          #
# --------------------------------------------------------------------------- #
class EmojiPrior:
    """
    Learns  log P(sarcastic | emoji) / P(non-sarcastic | emoji)  with add-k smoothing
    on the training split only.  At prediction time a post receives the mean / max /
    min / sum log-odds of its emojis.  This captures corpus-specific pragmatics
    (in this data ❤ and 🌹 almost never occur in sarcastic posts, 😉 and 😅 often do).
    """

    FEATURE_NAMES = ["emo_prior_mean", "emo_prior_max", "emo_prior_min", "emo_prior_sum"]

    def __init__(self, k: float = 1.0, min_count: int = 3):
        self.k, self.min_count = k, min_count
        self.logodds: Dict[str, float] = {}
        self.base = 0.0

    def fit(self, texts, labels):
        pos, neg = Counter(), Counter()
        n_pos = n_neg = 0
        for t, y in zip(texts, labels):
            es = {canonical(s["emoji"]) for s in extract(t)}
            if int(y) == 1:
                n_pos += 1
                pos.update(es)
            else:
                n_neg += 1
                neg.update(es)
        self.base = math.log((n_pos + self.k) / (n_neg + self.k))
        for e in set(pos) | set(neg):
            if pos[e] + neg[e] >= self.min_count:
                self.logodds[e] = math.log((pos[e] + self.k) / (neg[e] + self.k)) - self.base
        return self

    def transform_one(self, text: str) -> List[float]:
        vals = [self.logodds.get(canonical(s["emoji"]), 0.0) for s in extract(text)]
        if not vals:
            return [0.0, 0.0, 0.0, 0.0]
        return [sum(vals) / len(vals), max(vals), min(vals), sum(vals)]

    def transform(self, texts):
        return [self.transform_one(t) for t in texts]

    def top(self, n: int = 15):
        items = sorted(self.logodds.items(), key=lambda kv: kv[1])
        return {"most_non_sarcastic": items[:n], "most_sarcastic": items[-n:][::-1]}


if __name__ == "__main__":  # quick self-test
    s = "واہ جی واہ کیا بات ہے 😂😂😂 🙄"
    print(explain(s))
    print(emoji_to_tokens(s, "both"))
    print(emoji_to_tokens(s, "urdu"))
    print(dict(zip(EMOJI_FEATURE_NAMES, emoji_features(s))))
