def vowel_consonant_check(k): 
    vowel_Count=0 
    consonant_Count=0 
    for i in k:
        if i in ['a','e','i','o','u','A','E','I','O','U']: # checking for vowels
            vowel_Count+=1
        if i.upper() not in ['a','e','i','o','u','A','E','I','O','U'] and (ord(i.upper())>=65 and ord(i.upper())<=90): #checking for consonants and additional checks must be made to make sure that  special characters as not counted as  consonants
            consonant_Count+=1
    return vowel_Count, consonant_Count
k=input("Enter the string:")
vowel_count, consonant_count = vowel_consonant_check(k)
print("Number of vowels in the string is:",vowel_count)
print("Number of consonants in the string is:",consonant_count)


        
