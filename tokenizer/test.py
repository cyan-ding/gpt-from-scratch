
# we could use unicode to get the code point for every character to serve as tokens. but the vocabulary size would kinda be too big
# and we wouln't be able to do thing such as have one token for five spaces, which saves context size. 
# test = "hello world"
# print([ord(x) for x in test])

# instead, what we can do is take each of these unicode code points and convert them to bytes. 
# this is good because these byte encoders don't really change that much, unlike the uncide code point vocabulary


# the point of byte pair encoding (BPE) is so that you can compress large sequences
# the way it works is you find common pairs, and mint new tokens to represent those pairs
# and you do that recursively until you can't find common pairs anymore
# at that point you have a larger vocabulary but much smaller token sequence to feed into the LLM. 

from typing import Any


text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

vocab_size = 276 # number of merges, if we had 256 tokens originally, there would be 20 merges done 
num_merges = vocab_size - 256


def merge(ids, pair, idx):
    """
    ids = list of inputs
    pair = target pair to replace
    idx = number to replace with
    """

    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2 # skip over two indices
        else:
            newids.append(ids[i])
            i += 1
    return newids

    # print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))


def get_stats(bytes):
    # loops through all existing byte pairs and counts them
    counts = {}
    for pair in zip(bytes, bytes[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


ids = list[Any](tokens) # copy list of tokens
merges = {} # keep track of which tokens were merged into which 

for i in range(num_merges):
    stats = get_stats(ids) # get counts of each byte pair
    pair = max(stats, key=stats.get) # find largest pair by value (number of occurences)
    idx = 256 + i # create new token value
    print(f"merging pair {pair} into a new token {idx}")
    ids = merge(ids, pair, idx) # replace all instances of the pair with the new token
    merges[pair] = idx 

# this is my approach to decoding, basically create a reverse merge function that:
# given a idx demerge into a pair
# then, call this on the merges starting from the most recent merge.
# problem is, this costs O(mn), where m is number of merges and n is number of ids


# def reverse_merge(ids, idx, pair):
#     newids = []
#     i = 0

#     while i < len(ids):
#         if ids[i] == idx:
#             newids.extend(pair)
#         else:
#             newids.append(ids[i])
#         i += 1
#     return newids

# def decode(ids):
#     # given ids, return python string
#     # so we will probably need to do some binary tree kind of thing that will traverse the merges dictionary 
#     # we will need to take each of the unicode code points and convert them back using the inverse of the ord() function
#     # shouldn't be too hard...
#     copy_ids = list(ids)

#     for pair in reversed(merges):
#         copy_ids = reverse_merge(copy_ids, merges[pair], pair)

#     return bytes(copy_ids).decode("utf-8")

# print(decode(ids))

# ____________________________
# this is karpathy's approach 

vocab = { idx: bytes([idx]) for idx in range(256)} # create a dict that maps ints to byte versions
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1] # iterate through merges, setting the newly minted tokens' values to be the bytes of the merged pair

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids) # basically just create a bytestream with the new tokens converted back to the sum of 
    text = tokens.decode("utf-8", errors="replace")
    return text


print(decode([128]))







