# Description: Designing a simple blockchain with proof-of-work and transaction functionality.
# Key Concepts: Cryptography, data structures, proof-of-work

import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def proof_of_work(self, difficulty):
        target = '0' * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4

    def create_genesis_block(self):
        return Block(0, "0", "Genesis Block")

    def add_block(self, transactions):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, transactions)
        new_block.proof_of_work(self.difficulty)
        self.chain.append(new_block)

blockchain = Blockchain()
blockchain.add_block("Transaction1")
blockchain.add_block("Transaction2")
for block in blockchain.chain:
    print(f"Index: {block.index}, Hash: {block.hash}, Transactions: {block.transactions}")
