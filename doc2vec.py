def doc2vec():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # Not tags
    recipe_tags = recipes.drop(['title', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
    mean_rating = recipes['rating'].mean()
    # Creating new column where ratings are 1 for good and 0 for bad
    recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)
    ingred_to_rating = pd.concat((recipes['ingredients'], recipes['rating'], recipes['target']), axis=1, keys=['ingredients', 'rating', 'target'])
    print(ingred_to_rating.shape)
    #ingred_to_rating['ingredients'] = ingred_to_rating.ingredients.apply(lambda x: ' '.join(str(x)))


    print(ingred_to_rating['ingredients'].apply(lambda x: len(x.split(' '))).sum())
    print(ingred_to_rating.head())
    ingred_to_rating['ingredients'] = ingred_to_rating['ingredients'].apply(cleanText)
    train, test = train_test_split(ingred_to_rating, test_size=0.3, random_state=42)
    train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['ingredients']), tags=[r.rating]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['ingredients']), tags=[r.rating]), axis=1)





