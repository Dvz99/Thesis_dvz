study_case = [Noord, Kralingen-Crooswijk, Delfshaven, Centrum, Charlois, Feijenoord, IJsselmonde]

study_case_raw: [study_case], network_type = 'drive', simplify=False

study_case_strongly: same but with unreachable nodes removed (ox.truncate.largest_component(G, strongly=True))

clients_raw: all amenities with tags = {'office':True,'shop':True, 'amenity':['restaurant','bar','cafe','fast_food','food_court','pub','school']} (3774 points)
clients_nopoly: same but with polygon clients removed (3573 points)

