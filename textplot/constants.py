#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONNECTOR_WORDS = {
    'en': set([
        'without', 'a', 'to', 'by', 'of', 'and', 'at', 'or', 'for', 'an', 'from', 'in', 'on', 'the', 'with',
        'above', 'across', 'after', 'against', 'along', 'among', 'around', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'during', 'except', 'inside', 'into', 'near', 'onto', 'outside', 'over', 'through', 'under', 'upon',
        'but', 'nor', 'so', 'yet',
        'as', 'because', 'if', 'since', 'that', 'though', 'unless', 'until', 'when', 'where', 'while',
        'this', 'that', 'these', 'those', 'such', 'which', 'who', 'whom', 'whose'
    ]),
    'de': set([
        'ohne', 'ein', 'eine', 'zu', 'bei', 'von', 'und', 'an', 'in', 'oder', 'für', 'aus', 'auf', 'der', 'die', 'das', 'mit',
        'über', 'durch', 'nach', 'gegen', 'entlang', 'unter', 'um', 'vor', 'hinter', 'zwischen', 'während', 'außer', 'innerhalb', 'nahe', 'außerhalb', 'ausserhalb'
        'aber', 'noch', 'so', 'doch',
        'als', 'weil', 'wenn', 'seit', 'da', 'dass', 'obwohl', 'es sei denn', 'bis', 'wo', 'solange', 'falls', 'obgleich', 'unterhalb', 'neben', 'jenseits',
        'dieser', 'dieses', 'diese', 'jene', 'solcher', 'solches', 'solche', 'welcher', 'welches', 'welche', 'wessen', 'deren', 'dessen'
    ]),
    'fr': set([
        'sans', 'un', 'une', 'à', 'par', 'de', 'et', 'ou', 'pour', 'dans', 'en', 'sur', 'le', 'la', 'les', 'avec',
        'au-dessus de', 'à travers', 'après', 'contre', 'le long de', 'parmi', 'autour de', 'avant', 'derrière', 'au-dessous de', 'sous', 'à côté de', 'entre', 'au-delà de', 'pendant', 'sauf', 'à l\'intérieur de', 'près de', 'à l\'extérieur de',
        'mais', 'ni', 'donc', 'pourtant',
        'comme', 'parce que', 'si', 'depuis', 'que', 'bien que', 'à moins que', 'jusqu\'à', 'quand', 'où', 'pendant que',
        'ce', 'cet', 'cette', 'ces', 'cela', 'ça', 'tel', 'telle', 'tels', 'telles', 'qui', 'lequel', 'laquelle', 'lesquels', 'lesquelles', 'dont', 'chez'
    ]),
    'it': set([
        'senza', 'un', 'una', 'a', 'da', 'di', 'e', 'o', 'per', 'in', 'su', 'il', 'lo', 'la', 'i', 'gli', 'le', 'con',
        'sopra', 'attraverso', 'dopo', 'contro', 'lungo', 'tra', 'fra', 'intorno a', 'prima', 'dietro', 'sotto', 'accanto a', 'oltre', 'durante', 'eccetto', 'tranne', 'dentro', 'vicino a', 'fuori da',
        'ma', 'né', 'così',
        'come', 'perché', 'se', 'poiché', 'che', 'sebbene', 'a meno che', 'fino a', 'quando', 'dove', 'mentre',
        'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle', 'tale', 'tali', 'quale', 'quali', 'chi', 'di cui', 'il cui', 'la cui', 'i cui', 'le cui'
    ])
}
