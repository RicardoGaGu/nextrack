import math
class CamelotWheel:
    def __init__(self):
        self.essentiaToCamelotNotation = {'Ab minor':'1A','G# minor':'1A','B major':'1B','Eb minor':'2A','F# major':'2B','Gb major':'2B','Bb minor':'3A','C# major':'3B','Db major':'3B',
                    'F minor':'4A','Ab major':'4B','C minor':'5A','Eb major':'5B','G minor':'6A','Bb major':'6B',
                   'D minor':'7A','F major':'7B','A minor':'8A','C major':'8B','E minor':'9A','G major':'9B',
                   'B minor':'10A','D major':'10B','F# minor':'11A','A major':'11B','Db minor':'12A','C# minor':'12A','E major':'12B'}
        self.wheel = {}
        for n, key_letter in zip([x + 1 for x in range(24)], ['A', 'B'] * 12):
            # This makes it 1, 1, 2, 2, ..., 11, 11, 12, 12
            key_number = math.ceil(n / 2)
            # Create a set with the current key's harmonic neighbors. For example -> 1A: {1B, 2A, 12A}
            harmonic_neighbors = {
                f'{key_number}{key_letter}',
                f'{key_number}{self.__complement(key_letter)}',
                f'{self.__next(key_number)}{key_letter}',
                f'{self.__previous(key_number)}{key_letter}'
            }
            # Add entry to wheel dict
            self.wheel[f'{key_number}{key_letter}'] = harmonic_neighbors

    def __next(self,key_number):
        return key_number + 1 if key_number < 12 else 1
    def __previous(self, key_number):
        return key_number - 1 if key_number > 1 else 12
    def __complement(self, key_letter):
        return {'A': 'B', 'B': 'A'}[key_letter]
    def similar_keys(self,k1,k2):
        if k2 in self.wheel[k1]:
            return True
        else:
            return False
    def num_hops():
        return None
    # TODO
