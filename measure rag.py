import evaluate



# If you have a translation and reference corpus:
predictions = ["In Dungeons & Dragons, the metallic dragons come in brass, bronze, copper, gold, and silver varieties. Each has scales in hues matching their name - brass dragons have brassy scales, bronze have bronze scales, etc. The metallic dragons are generally more benign than the chromatic dragons in D&D lore."]


references =["""The five basic chromatic dragons (red, blue, green, black, and white) and metallic dragons (copper, brass, silver, gold, and bronze) appeared in the fifth edition Monster Manual (2014) in wyrmling, young, adult, and ancient. Gem dragons and other new-to-fifth-edition dragons appeared in Fizban's Treasury of Dragons (2021)"""]

rouge = evaluate.load('rouge')

results = rouge.compute(predictions=predictions, references=references, use_aggregator=False)
print(results)