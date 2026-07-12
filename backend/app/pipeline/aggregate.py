"""Turns raw per-frame detections into a biodiversity summary.

Two things happen here that the raw detection list can't give you on its own:
  - duplicate-count reduction: a track (one animal followed across frames)
    counts once per species, not once per frame it appeared in
  - cross-model dedup: when multiple models detect the same physical fish
    on the same frame, only the highest-confidence detection survives
  - low-confidence review: a track that never crossed its model's accept
    threshold in any frame goes into a review queue instead of being counted
"""
from collections import Counter, defaultdict
from itertools import combinations

COMMON_NAMES: dict[str, str] = {
    # FishInv / MegaFauna model classes
    "fish": "Unidentified Fish",
    "serranidae": "Grouper",
    "urchin": "Sea Urchin",
    "scaridae": "Parrotfish",
    "chaetodontidae": "Butterflyfish",
    "giant_clam": "Giant Clam",
    "lutjanidae": "Snapper",
    "muraenidae": "Moray Eel",
    "sea_cucumber": "Sea Cucumber",
    "haemulidae": "Grunt",
    "lobster": "Lobster",
    "crown_of_thorns": "Crown-of-Thorns Starfish",
    "bolbometopon_muricatum": "Bumphead Parrotfish",
    "cheilinus_undulatus": "Napoleon Wrasse",
    "cromileptes_altivelis": "Humpback Grouper",
    "ray": "Ray",
    "shark": "Shark",
    "turtle": "Sea Turtle",
    # Seychelles model classes (364 species)
    "Abudefduf natalensis": "Natal Sergeant",
    "Abudefduf septemfasciatus": "Banded Sergeant",
    "Abudefduf sexfasciatus": "Scissortail Sergeant",
    "Abudefduf sordidus": "Blackspot Sergeant",
    "Abudefduf sparoides": "False-eye Sergeant",
    "Abudefduf vaigiensis": "Indo-Pacific Sergeant",
    "Acanthocybium-solandri": "Wahoo",
    "Acanthurus blochii": "Ringtail Surgeonfish",
    "Acanthurus dussumieri": "Eyestripe Surgeonfish",
    "Acanthurus leucosternon": "Powder Blue Tang",
    "Acanthurus lineatus": "Lined Surgeonfish",
    "Acanthurus mata": "Elongate Surgeonfish",
    "Acanthurus nigricauda": "Epaulette Surgeonfish",
    "Acanthurus tennenti": "Lieutenant Surgeonfish",
    "Acanthurus thompsoni": "Thompson's Surgeonfish",
    "Acanthurus triostegus": "Convict Surgeonfish",
    "Acanthurus xanthopterus": "Yellowfin Surgeonfish",
    "Acathurus thompsoni": "Thompson's Surgeonfish",
    "Aethaloperca rogaa": "Redmouth Grouper",
    "Aetobatus narinari": "Spotted Eagle Ray",
    "Alectis ciliaris": "African Pompano",
    "Amblyeleotris aurora": "Pinkbar Goby",
    "Amblyglyphidodon batunai": "Batuna's Damselfish",
    "Amblyglyphidodon flavilatus": "Yellowfin Damselfish",
    "Amblyglyphidodon indicus": "Maldives Damselfish",
    "Amblygobius semicinctus": "Half-barred Goby",
    "Amphiprion akallopsis": "Skunk Anemonefish",
    "Amphiprion allardi": "Allard's Anemonefish",
    "Amphiprion bicinctus": "Twoband Anemonefish",
    "Amphiprion fuscocaudatus": "Seychelles Anemonefish",
    "Amphiprion latifasciatus": "Madagascar Anemonefish",
    "Anampses meleagrides": "Yellowtail Wrasse",
    "Anampses twistii": "Yellowbreasted Wrasse",
    "Aphareus furca": "Small-toothed Jobfish",
    "Apogon aureus": "Ring-tailed Cardinalfish",
    "Apogon kallopterus": "Iridescent Cardinalfish",
    "Apolemichthys trimaculatus": "Threespot Angelfish",
    "Arothron hispidus": "White-spotted Pufferfish",
    "Arothron mappa": "Map Pufferfish",
    "Arothron meleagris": "Guineafowl Pufferfish",
    "Arothron nigropunctatus": "Blackspotted Pufferfish",
    "Aspidontus dussumieri": "Lance Blenny",
    "Atherinomorus lacunosus": "Hardyhead Silverside",
    "Aulostomus chinensis": "Chinese Trumpetfish",
    "Balistapus undulatus": "Orange-lined Triggerfish",
    "Balistoides conspicillum": "Clown Triggerfish",
    "Balistoides viridescens": "Titan Triggerfish",
    "Black surgeonfish": "Black Surgeonfish",
    "Blenniella periophthalmus": "Blue-dashed Rockskipper",
    "Bodianus anthioides": "Lyretail Hogfish",
    "Bodianus axillaris": "Axilspot Hogfish",
    "Bodianus bilunulatus": "Saddleback Hogfish",
    "Bodianus diana": "Diana's Hogfish",
    "Bulbometopon muricatum": "Bumphead Parrotfish",
    "Caesio caerulaura": "Scissor-tailed Fusilier",
    "Caesio lunaris": "Lunar Fusilier",
    "Caesio varilineata": "Variable-lined Fusilier",
    "Caesio xanthonota": "Yellowback Fusilier",
    "Cantherhines pardalis": "Honeycomb Filefish",
    "Canthigaster bennetti": "Bennett's Toby",
    "Canthigaster papua": "Papuan Toby",
    "Canthigaster solandri": "Spotted Toby",
    "Canthigaster valentini": "Valentini's Sharpnose Pufferfish",
    "Carangoides caeruleopinnatus": "Coastal Trevally",
    "Caranx ignobilis": "Giant Trevally",
    "Caranx lugubris": "Black Jack",
    "Caranx melampygus": "Bluefin Trevally",
    "Caranx papuensis": "Brassy Trevally",
    "Caranx sexfasciatus": "Bigeye Trevally",
    "Carcharhinus albimarginatus": "Silvertip Shark",
    "Carcharhinus amblyrhynchos": "Grey Reef Shark",
    "Carcharhinus leucas": "Bull Shark",
    "Carcharinas melanopterus": "Blacktip Reef Shark",
    "Centropyge acanthops": "Orangeback Angelfish",
    "Centropyge multispinus": "Many-spined Angelfish",
    "Cephalopholis argus": "Peacock Grouper",
    "Cephalopholis boenak": "Chocolate Hind",
    "Cephalopholis leopardus": "Leopard Hind",
    "Cephalopholis miniata": "Coral Hind",
    "Cephalopholis nigripinnis": "Blackfin Grouper",
    "Cephalopholis urodeta": "Flagtail Grouper",
    "Chaetodon auriga": "Threadfin Butterflyfish",
    "Chaetodon bennetti": "Bennett's Butterflyfish",
    "Chaetodon dolosus": "African Butterflyfish",
    "Chaetodon falcula": "Blackwedged Butterflyfish",
    "Chaetodon guttatissimus": "Peppered Butterflyfish",
    "Chaetodon interruptus": "Yellow Teardrop Butterflyfish",
    "Chaetodon kleinii": "Klein's Butterflyfish",
    "Chaetodon lineatus": "Lined Butterflyfish",
    "Chaetodon lunula": "Raccoon Butterflyfish",
    "Chaetodon madagaskariensis": "Seychelles Butterflyfish",
    "Chaetodon melannotus": "Blackback Butterflyfish",
    "Chaetodon meyeri": "Meyer's Butterflyfish",
    "Chaetodon meyersi": "Meyer's Butterflyfish",
    "Chaetodon trifascialis": "Chevron Butterflyfish",
    "Chaetodon trifasciatus": "Melon Butterflyfish",
    "Chaetodon vagabundus": "Vagabond Butterflyfish",
    "Chaetodon xanthocephalus": "Yellowhead Butterflyfish",
    "Chaetodon zanzibariensis": "Zanzibar Butterflyfish",
    "Cheilinus fasciatus": "Red-breasted Wrasse",
    "Cheilinus trilobatus": "Tripletail Wrasse",
    "Cheilinus undulatus": "Humphead Wrasse",
    "Cheilio inermis": "Cigar Wrasse",
    "Cheilodipterus macrodon": "Large-toothed Cardinalfish",
    "Cheilodipterus quinquelineatus": "Five-lined Cardinalfish",
    "Chlorurus atrilunula": "Black Crescent Parrotfish",
    "Chlorurus sordidus": "Bullethead Parrotfish",
    "Chlorurus strongylocephalus": "Steephead Parrotfish",
    "Chromis dasygenys": "Damselfish",
    "Chromis dimidiata": "Chocolatedip Chromis",
    "Chromis nigura": "Blacktail Chromis",
    "Chromis opercularis": "Doublebar Chromis",
    "Chromis ternatensis": "Ternate Chromis",
    "Chromis viridis": "Blue-green Chromis",
    "Chromis weberi": "Weber's Chromis",
    "Chrysiptera biocellata": "Twinspot Damselfish",
    "Chrysiptera brownriggii": "Surge Damselfish",
    "Chrysiptera unimaculata": "Onespot Damselfish",
    "Cirrhilabrus exquisitus": "Exquisite Wrasse",
    "Cirrhitichthys oxycephalus": "Coral Hawkfish",
    "Cirrhitus pinnulatus": "Stocky Hawkfish",
    "Coris aygula": "Clown Coris",
    "Coris formosa": "Queen Coris",
    "Cryptocentrus cryptocentrus": "Shrimpgoby",
    "Ctenochaetus binotatus": "Twospot Bristletooth",
    "Ctenochaetus striatus": "Striated Surgeonfish",
    "Ctenochaetus strigosus": "Goldring Surgeonfish",
    "Dascyllus aruanus": "Humbug Dascyllus",
    "Dascyllus reticulatus": "Reticulate Dascyllus",
    "Dascyllus trimaculatus": "Threespot Dascyllus",
    "Dermatolepis striolata": "Smooth Grouper",
    "Diodon liturosus": "Black-blotched Porcupinefish",
    "Echeneis naucrates": "Live Sharksucker",
    "Ecsenius nalolo": "Nalolo Blenny",
    "Elagatis bipinnulatus": "Rainbow Runner",
    "Epibulus insidiator": "Sling-jaw Wrasse",
    "Epinephelus Tukula": "Potato Grouper",
    "Epinephelus andersoni": "Catface Grouper",
    "Epinephelus caeruleopunctatus": "Whitespotted Grouper",
    "Epinephelus fasciatus": "Blacktip Grouper",
    "Epinephelus flavocaeruleus": "Blue-and-yellow Grouper",
    "Epinephelus fuscoguttatus": "Brown-marbled Grouper",
    "Epinephelus lanceolatus": "Giant Grouper",
    "Epinephelus longispinis": "Longspine Grouper",
    "Epinephelus macrospilos": "Snubnose Grouper",
    "Epinephelus magniscuttis": "Grouper",
    "Epinephelus marginatus": "Dusky Grouper",
    "Epinephelus melanostigma": "One-blotch Grouper",
    "Epinephelus merra": "Honeycomb Grouper",
    "Epinephelus multinotatus": "White-blotched Grouper",
    "Epinephelus polyphekadion": "Camouflage Grouper",
    "Epinephelus rivulatus": "Halfmoon Grouper",
    "Epinephelus spilotoceps": "Foursaddle Grouper",
    "Epinephelus tauvina": "Greasy Grouper",
    "Escenius midas": "Midas Blenny",
    "Euthynnus affinis": "Kawakawa",
    "Forcipiger flavissimus": "Longnose Butterflyfish",
    "Galeocerdo cuvier": "Tiger Shark",
    "Gerres longirostris": "Strongspine Silver-biddy",
    "Gnathanodon speciosus": "Golden Trevally",
    "Gnathodentex aureolineatus": "Striped Large-eye Bream",
    "Gnatholepis cauerensis": "Eyebar Goby",
    "Gomphosus caeruleus": "Indian Ocean Bird Wrasse",
    "Gracila albomarginata": "Masked Grouper",
    "Grammistes sexlineatus": "Sixline Soapfish",
    "Gymnocranius griseus": "Grey Large-eye Bream",
    "Gymnosarda unicolor": "Dogtooth Tuna",
    "Gymnothorax breedeni": "Blackcheek Moray",
    "Gymnothorax favagineus": "Honeycomb Moray",
    "Gymnothorax javanicus": "Giant Moray",
    "Gymnothorax meleagris": "Whitemouth Moray",
    "Gymnothorax pictus": "Peppered Moray",
    "Gymnothorax undulatus": "Undulated Moray",
    "Halichoeres cosmetus": "Adorned Wrasse",
    "Halichoeres hortulanus": "Checkerboard Wrasse",
    "Halichoeres nebulosus": "Nebulous Wrasse",
    "Halichoeres scapularis": "Zigzag Wrasse",
    "Hemigymnus fasciatus": "Barred Thicklip Wrasse",
    "Hemigymnus melapterus": "Blackeye Thicklip Wrasse",
    "Hemitaurichthys zoster": "Black Pyramid Butterflyfish",
    "Heniochus acuminatus": "Longfin Bannerfish",
    "Heniochus diphreutes": "Schooling Bannerfish",
    "Heniochus monoceros": "Masked Bannerfish",
    "Heteroconger hassi": "Spotted Garden Eel",
    "Himantura fai": "Pink Whipray",
    "Hipposcarus harid": "Longnose Parrotfish",
    "Hologymnosus annulatus": "Ring Wrasse",
    "Istiophorus platypterus": "Indo-Pacific Sailfish",
    "Kuhlia mugil": "Barred Flagtail",
    "Kyphosus bigibbus": "Brown Chub",
    "Labroides dimidiatus": "Bluestreak Cleaner Wrasse",
    "Lethrinus Mahsena": "Sky Emperor",
    "Lethrinus conchyliatus": "Redaxil Emperor",
    "Lethrinus harak": "Thumbprint Emperor",
    "Lethrinus obsoletus": "Orange-striped Emperor",
    "Lethrinus rubioperculatus": "Spotcheek Emperor",
    "Lethrinus semicinctus": "Emperor",
    "Lethrinus xanthochilus": "Yellowlip Emperor",
    "Lutjanus argentimaculatus": "Mangrove Red Snapper",
    "Lutjanus bohar": "Two-spot Red Snapper",
    "Lutjanus ehrenbergii": "Blackspot Snapper",
    "Lutjanus fulviflamma": "Dory Snapper",
    "Lutjanus fulvus": "Blacktail Snapper",
    "Lutjanus gibbus": "Humpback Red Snapper",
    "Lutjanus kasmira": "Common Bluestripe Snapper",
    "Lutjanus monostigma": "One-spot Snapper",
    "Lutjanus rivulatus": "Blubberlip Snapper",
    "Macolor niger": "Black and White Snapper",
    "Macropharyngodon bipartitus": "Vermiculate Wrasse",
    "Malacanthus atovittatus": "Striped Blanquillo",
    "Malacanthus brevirostris": "Quakerfish",
    "Malacanthus latovittatus": "Striped Blanquillo",
    "Manta birostris": "Giant Oceanic Manta Ray",
    "Meiacanthus mossambicus": "Mozambique Fangblenny",
    "Melichthys niger": "Black Triggerfish",
    "Monodactylus argenteus": "Silver Moony",
    "Monotaxis grandoculis": "Humpnose Big-eye Bream",
    "Mulloides vanicolensis": "Yellowfin Goatfish",
    "Mulloidichthys flavolineatus": "Yellowstripe Goatfish",
    "Myrichthys colubrinus": "Harlequin Snake Eel",
    "Myripristis botche": "Blacktip Soldierfish",
    "Myripristis kuntee": "Shoulderbar Soldierfish",
    "Myripristis murdjan": "Pinecone Soldierfish",
    "Naso annulatus": "Whitemargin Unicornfish",
    "Naso brevicordatus": "Unicornfish",
    "Naso hexacanthus": "Sleek Unicornfish",
    "Naso lituratus": "Orangespine Unicornfish",
    "Naso tuberosus": "Humpnose Unicornfish",
    "Naso unicornis": "Bluespine Unicornfish",
    "Nebrius ferrugineus": "Tawny Nurse Shark",
    "Negaprion acutidens": "Sicklefin Lemon Shark",
    "Nemateleotris magnifica": "Fire Dartfish",
    "Neoglyphidodon bonang": "Ocellate Damselfish",
    "Neoglyphidodon melas": "Bowtie Damselfish",
    "Neoniphon sammara": "Sammara Squirrelfish",
    "Neopomacentrus sororius": "Demoiselle",
    "Neotrygon kuhlii": "Kuhl's Stingray",
    "Novaculichthys taeniourus": "Rockmover Wrasse",
    "Odonus niger": "Red-toothed Triggerfish",
    "Osctracion cubicus": "Yellow Boxfish",
    "Ostorhinchus apogonoides": "Short-tooth Cardinalfish",
    "Ostorhinchus lateralis": "Humpback Cardinalfish",
    "Ostorhinchus taeniophorus": "Reef-flat Cardinalfish",
    "Ostracion melagris": "Whitespotted Boxfish",
    "Oxycheilinus digramma": "Cheeklined Wrasse",
    "Oxymoncanthus longirostris": "Longnose Filefish",
    "Paracanthurus hepatus": "Palette Surgeonfish",
    "Paracirrhites arcatus": "Arc-eye Hawkfish",
    "Paracirrhites forsteri": "Freckled Hawkfish",
    "Paraluteres prionurus": "Blacksaddle Filefish",
    "Parapercis hexophtalma": "Speckled Sandperch",
    "Parapercis punctulata": "Sandperch",
    "Parupeneus barberinus": "Dash-and-dot Goatfish",
    "Parupeneus bifasciatus": "Doublebar Goatfish",
    "Parupeneus ciliatus": "Whitesaddle Goatfish",
    "Parupeneus cyclostomus": "Gold-saddle Goatfish",
    "Parupeneus indicus": "Indian Goatfish",
    "Parupeneus macronema": "Long-barbel Goatfish",
    "Parupeneus trifasciatus": "Indian Doublebar Goatfish",
    "Pempheris adusta": "Dusky Sweeper",
    "Pempheris oualensis": "Silver Sweeper",
    "Periophthalmus kalolo": "Mudskipper",
    "Plagiotremus rhinorhynchos": "Bluestriped Fangblenny",
    "Plagiotremus tapeinosoma": "Piano Fangblenny",
    "Platax orbicularis": "Orbicular Batfish",
    "Platax teira": "Longfin Batfish",
    "Plectorhichus flavomaculatus": "Gold-spotted Sweetlips",
    "Plectorhinchus albovittatu": "Two-striped Sweetlips",
    "Plectorhinchus gaterinus": "Blackspotted Sweetlips",
    "Plectorhinchus gibbosus": "Harry Hotlips",
    "Plectorhinchus paulayi": "Sweetlips",
    "Plectorhinchus plagiodesmus": "Sweetlips",
    "Plectorhinchus playfairi": "Whitebarred Rubberlip",
    "Plectorhinchus sordidus": "Sordid Sweetlips",
    "Plectorhinchus vittatus": "Oriental Sweetlips",
    "Plectroglyphidodon dickii": "Blackbar Devil",
    "Plectroglyphidodon imparipennis": "Brighteye Damselfish",
    "Plectroglyphidodon johnstonianus": "Johnston's Damselfish",
    "Plectroglyphidodon lacrymatus": "Jewel Damselfish",
    "Plectroglyphidodon leucozo": "Singlebar Devil",
    "Plectroglyphidon lacrymatus": "Jewel Damselfish",
    "Plectropomus laevis": "Blacksaddled Coral Grouper",
    "Plectropomus pessuliferus": "Roving Coral Grouper",
    "Plectropomus punctatus": "Marbled Coral Grouper",
    "Pomacanthus chrysurus": "Goldtail Angelfish",
    "Pomacanthus imperator": "Emperor Angelfish",
    "Pomacanthus rhomboides": "Old Woman Angelfish",
    "Pomacanthus semicirculatus": "Semicircle Angelfish",
    "Pomacentrus caeruleopunctatus": "Damselfish",
    "Pomacentrus caeruleus": "Caerulean Damselfish",
    "Pomacentrus pavo": "Sapphire Damselfish",
    "Pomacentrus sulfureus": "Sulphur Damselfish",
    "Pomacentrus trichourus": "Damselfish",
    "Pomadasys furcatum": "Grunt",
    "Priacanthus hamrur": "Moontail Bullseye",
    "Pseadanthias cooperi": "Cooper's Anthias",
    "Pseadanthias squamipinnis": "Sea Goldie",
    "Pseudanthias evansi": "Evan's Anthias",
    "Pseudanthias taeniatus": "Striped Anthias",
    "Pseudocaranx dentex": "White Trevally",
    "Pseudocheilinus evanidus": "Striated Wrasse",
    "Pseudodax mollucanus": "Chiseltooth Wrasse",
    "Ptereleotris evides": "Blackfin Dartfish",
    "Pterocaesio digramma": "Double-lined Fusilier",
    "Pterocaesio marri": "Marr's Fusilier",
    "Pterocaesio randalli": "Randall's Fusilier",
    "Pterocaesio tessellata": "One-stripe Fusilier",
    "Pterocaesio tile": "Dark-banded Fusilier",
    "Pterocaesio trilineata": "Three-stripe Fusilier",
    "Pterois miles": "Devil Firefish",
    "Pterois muricata": "Lionfish",
    "Pygoplites diacanthus": "Regal Angelfish",
    "Rhinecanthus aculeatus": "Picasso Triggerfish",
    "Rhinecanthus rectangulus": "Wedge-tail Triggerfish",
    "Sargocentron caudimaculatum": "Silverspot Squirrelfish",
    "Sargocentron diadema": "Crown Squirrelfish",
    "Sargocentron spiniferum": "Sabre Squirrelfish",
    "Scarus caudofasciatus": "Parrotfish",
    "Scarus ghobban": "Blue-barred Parrotfish",
    "Scarus oviceps": "Dark-capped Parrotfish",
    "Scarus persicus": "Persian Parrotfish",
    "Scarus psittacus": "Common Parrotfish",
    "Scarus rubroviolaceus": "Ember Parrotfish",
    "Scarus russelii": "Russell's Parrotfish",
    "Scarus scaber": "Fivesaddle Parrotfish",
    "Scarus tricolor": "Tricolor Parrotfish",
    "Scolopsis ghanam": "Arabian Monocle Bream",
    "Scombroides lysan": "Doublespotted Queenfish",
    "Scorpaenopsis cirrosa": "Tasseled Scorpionfish",
    "Sebastapistes cyanostigma": "Scorpionfish",
    "Seriola rivoliana": "Almaco Jack",
    "Siganus argenteus": "Forktail Rabbitfish",
    "Siganus luridus": "Dusky Rabbitfish",
    "Siganus stellatus": "Stellate Rabbitfish",
    "Sphaeramia orbicularis": "Orbiculate Cardinalfish",
    "Sphyraena barracuda": "Great Barracuda",
    "Sphyraena qenie": "Blackfin Barracuda",
    "Sphyrna lewini": "Scalloped Hammerhead",
    "Sprion virescens": "Reef Fish",
    "Stegostoma fasiatum": "Zebra Shark",
    "Stethojulis albovittata": "Bluelined Wrasse",
    "Sufflamen albicaudatum": "Triggerfish",
    "Sufflamen bursa": "Lei Triggerfish",
    "Sufflamen chrysopteros": "Halfmoon Triggerfish",
    "Sufflamen chrysopterus": "Halfmoon Triggerfish",
    "Synodus dermatogenys": "Sand Lizardfish",
    "Thalassoma amblycephalum": "Twotone Wrasse",
    "Thalassoma genivittatum": "Wrasse",
    "Thalassoma hardwicke": "Sixbar Wrasse",
    "Thalassoma hebraicum": "Goldbar Wrasse",
    "Thalassoma lunare": "Moon Wrasse",
    "Thalassoma trilobatum": "Christmas Wrasse",
    "Torpedo fuscomaculata": "Dark-spotted Electric Ray",
    "Trachinotus Baillonii": "Small-spotted Dart",
    "Trachinotus blochi": "Snubnose Pompano",
    "Triaenodon obesus": "Whitetip Reef Shark",
    "Tripterodon orbis": "African Spadefish",
    "Umbrina ronchus": "Croaker",
    "Urogymnus asperrimus": "Porcupine Ray",
    "Valamugil buchanini": "Buchanan's Mullet",
    "Valenciennea strigata": "Blueband Goby",
    "Zanclus canescens": "Moorish Idol",
    "Zebrasoma desjardinii": "Desjardin's Sailfin Tang",
    "Zebrasoma scopas": "Brown Tang",
    # Lionfish model class
    "Lionfish": "Lionfish",
    # Corals model classes (19 genera)
    "Acanthastrea": "Acanthastrea Coral",
    "Acropora": "Acropora Coral",
    "Coeloseris": "Coeloseris Coral",
    "Euphyllia": "Euphyllia Coral",
    "Favia": "Favia Coral",
    "Favites": "Favites Coral",
    "Goniastrea": "Goniastrea Coral",
    "Heterocyathus": "Heterocyathus Coral",
    "Isopora": "Isopora Coral",
    "Leptoseris": "Leptoseris Coral",
    "Millepora": "Fire Coral",
    "Pachyseris": "Pachyseris Coral",
    "Pocillopora": "Pocillopora Coral",
    "Porites": "Porites Coral",
    "Psammocora": "Psammocora Coral",
    "Sandalolitha": "Sandalolitha Coral",
    "Stylophora": "Stylophora Coral",
    "Trachyphilia": "Trachyphyllia Coral",
    "Turbinaria": "Turbinaria Coral",
    # MultiClass model classes
    "Blue-Tang": "Blue Tang",
    "Orange-Clown": "Clownfish",
    "Three-Striped-Damselfish": "Three-Stripe Damselfish",
    "Yellow-Tang": "Yellow Tang",
    # MarineLife model classes (filtered to reef-relevant)
    "eel": "Eel",
    "starfish": "Starfish",
    "crab": "Crab",
    "jellyfish": "Jellyfish",
    # ReefFamilies model classes (13 families)
    "Acanthuridae -Surgeonfishes-": "Surgeonfish",
    "Balistidae -Triggerfishes-": "Triggerfish",
    "Carangidae -Jacks-": "Jack",
    "Ephippidae -Spadefishes-": "Spadefish",
    "Labridae -Wrasse-": "Wrasse",
    "Lutjanidae -Snappers-": "Snapper",
    "Pomacanthidae -Angelfishes-": "Angelfish",
    "Pomacentridae -Damselfishes-": "Damselfish",
    "Scaridae -Parrotfishes-": "Parrotfish",
    "Scombridae -Tunas-": "Tuna",
    "Serranidae -Groupers-": "Grouper",
    "Shark -Selachimorpha-": "Shark",
    "Zanclidae -Moorish Idol-": "Moorish Idol",
    # FishSpecies model classes (481 species)
    "A73EGS-P_5": "Unknown Fish",
    "CUNWCB-Y": "Unknown Fish",
    "Istiophorus_platypterus": "Indo-Pacific Sailfish",
    "P1ROZC-Z": "Unknown Fish",
    "PQV7DP-S": "Unknown Fish",
    "acanthaluteres_brownii": "Brownii Leatherjacket",
    "acanthaluteres_spilomelanurus": "Spilomelanurus Leatherjacket",
    "acanthaluteres_vittiger": "Vittiger Leatherjacket",
    "acanthistius_cinctus": "Acanthistius Cinctus",
    "acanthopagrus_australis": "Australis Sea Bream",
    "acanthopagrus_berda": "Berda Sea Bream",
    "acanthopagrus_latus": "Latus Sea Bream",
    "achoerodus_gouldii": "Western Blue Groper",
    "achoerodus_viridis": "Eastern Blue Groper",
    "acreichthys_tomentosus": "Tomentosus Filefish",
    "aesopia_cornuta": "Cornuta Sole",
    "aethaloperca_rogaa": "Rogaa Redmouth Grouper",
    "alectis_ciliaris": "Ciliaris Pompano",
    "alectis_indica": "Indica Pompano",
    "alepes_kleinii": "Kleinii Scad",
    "aluterus_monoceros": "Monoceros Filefish",
    "aluterus_scriptus": "Scriptus Filefish",
    "amanses_scopas": "Scopas Filefish",
    "anampses_caeruleopunctatus": "Caeruleopunctatus Wrasse",
    "anampses_elegans": "Elegans Wrasse",
    "anampses_femininus": "Femininus Wrasse",
    "anampses_geographicu": "Geographicu Wrasse",
    "anampses_lennardi": "Lennardi Wrasse",
    "anampses_melanurus": "Melanurus Wrasse",
    "anampses_meleagrides": "Meleagrides Wrasse",
    "anampses_neoguinaicus": "Neoguinaicus Wrasse",
    "anampses_twistii": "Twistii Wrasse",
    "anodontostoma_chacunda": "Chacunda Gizzard Shad",
    "anyperodon_leucogrammicus": "Leucogrammicus Slender Grouper",
    "aphareus_furca": "Small-toothed Jobfish",
    "aphareus_rutilans": "Rusty Jobfish",
    "aprion_virescens": "Green Jobfish",
    "argyrops_spinifer": "Spinifer Sea Bream",
    "aseraggodes_melanostictus": "Melanostictus Sole",
    "atractoscion_aequidens": "Aequidens Geelbek",
    "atule_mate": "Mate Yellowtail Scad",
    "auxis_rochei": "Bullet Tuna",
    "auxis_thazard": "Frigate Tuna",
    "bathylagichthys_greyae": "Greyae Deep-sea Smelt",
    "beryx_decadactylus": "Decadactylus Alfonsino",
    "bodianus_anthioides": "Anthioides Hogfish",
    "bodianus_axillaris": "Axilspot Hogfish",
    "bodianus_bilunulatus": "Bilunulatus Hogfish",
    "bodianus_bimaculatus": "Bimaculatus Hogfish",
    "bodianus_diana": "Diana's Hogfish",
    "bodianus_loxozonus": "Loxozonus Hogfish",
    "bodianus_mesothorax": "Mesothorax Hogfish",
    "bodianus_perditio": "Perditio Hogfish",
    "bodianus_unimaculatus": "Unimaculatus Hogfish",
    "bodianus_vulpinus": "Vulpinus Hogfish",
    "bothus_mancus": "Mancus Flounder",
    "bothus_myriaster": "Myriaster Flounder",
    "bothus_pantherinus": "Pantherinus Flounder",
    "brachaluteres_jacksonianus": "Jacksonianus Pygmy Leatherjacket",
    "brachirus_orientalis": "Orientalis Sole",
    "caesioperca_lepidopterus": "Lepidopterus Butterfly Perch",
    "cantherhines_dumerilii": "Dumerilii Filefish",
    "cantherhines_fronticinctus": "Fronticinctus Filefish",
    "cantherhines_pardalis": "Pardalis Filefish",
    "cantheschenia_grandisquamis": "Grandisquamis Leatherjacket",
    "caprodon_longimanus": "Longimanus Perch",
    "caprodon_schlegelii": "Schlegelii Perch",
    "carangoides_caeruleopinnatus": "Caeruleopinnatus Trevally",
    "carangoides_chrysophrys": "Chrysophrys Trevally",
    "carangoides_equula": "Equula Trevally",
    "carangoides_ferdau": "Ferdau Trevally",
    "carangoides_fulvoguttatus": "Fulvoguttatus Trevally",
    "carangoides_hedlandensis": "Hedlandensis Trevally",
    "carangoides_malabaricus": "Malabaricus Trevally",
    "carangoides_orthogrammus": "Orthogrammus Trevally",
    "carangoides_plagiotaenia": "Plagiotaenia Trevally",
    "caranx_ignobilis": "Giant Trevally",
    "caranx_lugubris": "Black Jack",
    "caranx_melampygus": "Bluefin Trevally",
    "caranx_sexfasciatus": "Bigeye Trevally",
    "carcharhinus_albimarginatu": "Silvertip Shark",
    "carcharhinus_amblyrhynchos": "Grey Reef Shark",
    "carcharhinus_falciformis": "Silky Shark",
    "carcharhinus_galapagensis": "Galapagos Shark",
    "carcharhinus_limbatus": "Blacktip Shark",
    "carcharhinus_melanopterus": "Blacktip Reef Shark",
    "carcharhinus_obscurus": "Dusky Shark",
    "carcharhinus_plumbeus": "Sandbar Shark",
    "carcharhinus_sorrah": "Spottail Shark",
    "centroberyx_affinis": "Affinis Nannygai",
    "centrogenys_vaigiensis": "Vaigiensis False Scorpionfish",
    "centroscymnus_coelolepis": "Portuguese Dogfish",
    "cephalopholis_argus": "Peacock Grouper",
    "cephalopholis_boenak": "Boenak Grouper",
    "cephalopholis_cyanostigma": "Cyanostigma Grouper",
    "cephalopholis_formosa": "Formosa Grouper",
    "cephalopholis_igarashiensis": "Igarashiensis Grouper",
    "cephalopholis_leopardus": "Leopardus Grouper",
    "cephalopholis_microprion": "Microprion Grouper",
    "cephalopholis_miniata": "Coral Hind",
    "cephalopholis_sexmaculata": "Sexmaculata Grouper",
    "cephalopholis_sonnerati": "Sonnerati Grouper",
    "cephalopholis_spiloparaea": "Spiloparaea Grouper",
    "chascanopsetta_lugubris": "Lugubris Flounder",
    "cheilinus_chlorourus": "Chlorourus Wrasse",
    "cheilinus_fasciatus": "Fasciatus Wrasse",
    "cheilinus_oxycephalus": "Oxycephalus Wrasse",
    "cheilinus_trilobatus": "Trilobatus Wrasse",
    "cheilinus_undulatus": "Humphead Wrasse",
    "cheilio_inermis": "Inermis Cigar Wrasse",
    "cheilodactylus_ephippium": "Ephippium Morwong",
    "cheilodactylus_fuscus": "Fuscus Morwong",
    "cheilodactylus_vestitus": "Vestitus Morwong",
    "chelidonichthys_kumu": "Kumu Gurnard",
    "chirocentrus_dorab": "Dorab Wolf Herring",
    "chirocentrus_nudus": "Nudus Wolf Herring",
    "choerodon_anchorago": "Orange-dotted Tuskfish",
    "choerodon_cauteroma": "Cauteroma Tuskfish",
    "choerodon_cyanodus": "Cyanodus Tuskfish",
    "choerodon_fasciatus": "Harlequin Tuskfish",
    "choerodon_graphicus": "Graphicus Tuskfish",
    "choerodon_jordani": "Jordani Tuskfish",
    "choerodon_rubescens": "Rubescens Tuskfish",
    "choerodon_schoenleinii": "Blackspot Tuskfish",
    "choerodon_venustus": "Venustus Tuskfish",
    "choerodon_vitta": "Vitta Tuskfish",
    "choerodon_zamboangae": "Zamboangae Tuskfish",
    "chromileptes_altivelis": "Humpback Grouper",
    "cirrhilabrus_bathyphilus": "Bathyphilus Fairy Wrasse",
    "cirrhilabrus_condei": "Condei Fairy Wrasse",
    "cirrhilabrus_cyanopleura": "Cyanopleura Fairy Wrasse",
    "cirrhilabrus_exquisitus": "Exquisitus Fairy Wrasse",
    "cirrhilabrus_laboutei": "Laboutei Fairy Wrasse",
    "cirrhilabrus_punctatus": "Punctatus Fairy Wrasse",
    "cirrhilabrus_scottorum": "Scottorum Fairy Wrasse",
    "cirrhilabrus_temminckii": "Temminckii Fairy Wrasse",
    "coris_aygula": "Clown Coris",
    "coris_batuensis": "Batuensis Wrasse",
    "coris_bulbifrons": "Bulbifrons Wrasse",
    "coris_caudimacula": "Caudimacula Wrasse",
    "coris_dorsomacula": "Dorsomacula Wrasse",
    "coris_gaimard": "Gaimard Wrasse",
    "coris_picta": "Picta Wrasse",
    "coris_pictoides": "Pictoides Wrasse",
    "coris_sandeyeri": "Sandeyeri Wrasse",
    "crenimugil_crenilabis_2": "Crenilabis 2 Fringelip Mullet",
    "cymbacephalus_nematophthalmus": "Nematophthalmus Crocodile Flathead",
    "cymolutes_praetextatus": "Praetextatus Razorfish",
    "cymolutes_torquatus": "Torquatus Razorfish",
    "cynoglossus_puncticeps": "Puncticeps Tongue Sole",
    "cyttopsis_rosea": "Rosea Dory",
    "dactylophora_nigricans": "Nigricans Banjo Shark",
    "decapterus_macrosoma": "Macrosoma Mackerel Scad",
    "decapterus_russelli": "Russelli Mackerel Scad",
    "diproctacanthus_xanthurus": "Xanthurus Wrasse",
    "dotalabrus_aurantiacus": "Aurantiacus Wrasse",
    "elagatis_bipinnulata": "Rainbow Runner",
    "epibulus_insidiator": "Sling-jaw Wrasse",
    "epinephelus_areolatus": "Areolatus Grouper",
    "epinephelus_bleekeri": "Bleekeri Grouper",
    "epinephelus_chlorostigma": "Chlorostigma Grouper",
    "epinephelus_coeruleopunctatus": "Coeruleopunctatus Grouper",
    "epinephelus_coioides": "Orange-spotted Grouper",
    "epinephelus_corallicola": "Corallicola Grouper",
    "epinephelus_cyanopodus": "Cyanopodus Grouper",
    "epinephelus_epistictus": "Epistictus Grouper",
    "epinephelus_fasciatus": "Blacktip Grouper",
    "epinephelus_fuscoguttatus": "Brown-marbled Grouper",
    "epinephelus_hexagonatus": "Hexagonatus Grouper",
    "epinephelus_howlandi": "Howlandi Grouper",
    "epinephelus_lanceolatus": "Giant Grouper",
    "epinephelus_latifasciatus": "Latifasciatus Grouper",
    "epinephelus_macrospilos": "Macrospilos Grouper",
    "epinephelus_maculatus": "Maculatus Grouper",
    "epinephelus_melanostigma": "Melanostigma Grouper",
    "epinephelus_merra": "Honeycomb Grouper",
    "epinephelus_morrhua": "Morrhua Grouper",
    "epinephelus_multinotatus": "Multinotatus Grouper",
    "epinephelus_ongus": "Ongus Grouper",
    "epinephelus_polyphekadion": "Polyphekadion Grouper",
    "epinephelus_quoyanus": "Quoyanus Grouper",
    "epinephelus_radiatus": "Radiatus Grouper",
    "epinephelus_retouti": "Retouti Grouper",
    "epinephelus_rivulatus": "Rivulatus Grouper",
    "epinephelus_sexfasciatus": "Sexfasciatus Grouper",
    "epinephelus_spilotoceps": "Spilotoceps Grouper",
    "epinephelus_tauvina": "Tauvina Grouper",
    "epinephelus_undulatostriatus": "Undulatostriatus Grouper",
    "etelis_carbunculus": "Carbunculus Ruby Snapper",
    "etelis_coruscan": "Coruscan Ruby Snapper",
    "eubalichthys_cyanoura": "Cyanoura Leatherjacket",
    "eubalichthys_mosaicus": "Mosaicus Leatherjacket",
    "eupetrichthys_angustipes": "Angustipes Snakeblenny",
    "euthynnus_affinis": "Kawakawa",
    "evistias_acutirostris": "Acutirostris Striped Boarfish",
    "gempylus_serpens": "Serpens Snake Mackerel",
    "gnathanodon_speciosus": "Golden Trevally",
    "gnathodentex_aureolineatus": "Striped Large-eye Bream",
    "gracila_albomarginata": "Masked Grouper",
    "gymnocranius_audleyi": "Audleyi Large-eye Bream",
    "gymnocranius_euanus": "Euanus Large-eye Bream",
    "gymnocranius_grandoculis": "Grandoculis Large-eye Bream",
    "gymnocranius_microdon": "Microdon Large-eye Bream",
    "gymnosarda_unicolor": "Dogtooth Tuna",
    "halichoeres_argus": "Argus Wrasse",
    "halichoeres_biocellatus": "Biocellatus Wrasse",
    "halichoeres_chloropterus": "Chloropterus Wrasse",
    "halichoeres_chrysus": "Chrysus Wrasse",
    "halichoeres_hartzfeldii": "Hartzfeldii Wrasse",
    "halichoeres_hortulanus": "Hortulanus Wrasse",
    "halichoeres_leucurus": "Leucurus Wrasse",
    "halichoeres_margaritaceus": "Margaritaceus Wrasse",
    "halichoeres_melanochir": "Melanochir Wrasse",
    "halichoeres_melanurus": "Melanurus Wrasse",
    "halichoeres_melasmapomus": "Melasmapomus Wrasse",
    "halichoeres_nebulosus": "Nebulosus Wrasse",
    "halichoeres_nigrescens": "Nigrescens Wrasse",
    "halichoeres_scapularis": "Scapularis Wrasse",
    "halichoeres_trimaculatus": "Trimaculatus Wrasse",
    "harriotta_raleighana": "Pacific Longnose Chimaera",
    "hemigymnus_fasciatus": "Barred Thicklip Wrasse",
    "hemigymnus_melapterus": "Blackeye Thicklip Wrasse",
    "hemiramphus_far": "Black-barred Halfbeak",
    "hemiramplutjanus_semicinctus_quoyhus_far": "Hemiramplutjanus Semicinctus Quoyhus Far",
    "herklotsichthys_quadrimaculatus": "Quadrimaculatus Herring",
    "hologymnosus_annulatus": "Ring Wrasse",
    "hologymnosus_doliatus": "Pastel Ring Wrasse",
    "hyporhamphus_affinis": "Affinis Halfbeak",
    "hyporhamphus_dussumieri": "Dussumieri Halfbeak",
    "inegocia_japonica": "Japonica Flathead",
    "johnius_borneensis": "Borneensis Croaker",
    "katsuwonus_pelamis": "Skipjack Tuna",
    "labrichthys_unilineatus": "Unilineatus Tubelip Wrasse",
    "labroides_bicolor": "Bicolor Cleaner Wrasse",
    "labroides_dimidiatus": "Bluestreak Cleaner Wrasse",
    "labroides_pectoralis": "Pectoralis Cleaner Wrasse",
    "labropsis_australis": "Australis Wrasse",
    "labropsis_manabei": "Manabei Wrasse",
    "labropsis_xanthonota": "Xanthonota Wrasse",
    "latridopsis_forsteri": "Forsteri Trumpeter",
    "lepidocybium_flavobrunneum": "Escolar",
    "leptojulis_cyanopleura": "Cyanopleura Wrasse",
    "lethrinus_amboinensis": "Ambon Emperor",
    "lethrinus_atkinsoni": "Pacific Yellowtail Emperor",
    "lethrinus_erythracanthus": "Orange-spotted Emperor",
    "lethrinus_genivittatus": "Threadfin Emperor",
    "lethrinus_harak": "Thumbprint Emperor",
    "lethrinus_lentjan": "Pink-ear Emperor",
    "lethrinus_microdon": "Smalltooth Emperor",
    "lethrinus_miniatus": "Trumpet Emperor",
    "lethrinus_nebulosus": "Spangled Emperor",
    "lethrinus_obsoletus": "Orange-striped Emperor",
    "lethrinus_olivaceus": "Longface Emperor",
    "lethrinus_ornatus": "Ornate Emperor",
    "lethrinus_rubrioperculatus": "Spotcheek Emperor",
    "lethrinus_semicinctus": "Black Blotch Emperor",
    "lethrinus_variegatus": "Slender Emperor",
    "lethrinus_xanthochilus": "Yellowlip Emperor",
    "liopropoma_mitratum": "Mitratum Basslet",
    "liopropoma_susumi": "Susumi Basslet",
    "liza_subviridis": "Subviridis Mullet",
    "liza_vaigiensis": "Vaigiensis Mullet",
    "lniistius_aneitensis": "Aneitensis Razorfish",
    "lniistius_pavo": "Pavo Razorfish",
    "lutjanus_adetii": "Adetii Snapper",
    "lutjanus_argentimaculatus": "Mangrove Red Snapper",
    "lutjanus_biguttatus": "Biguttatus Snapper",
    "lutjanus_bohar": "Two-spot Red Snapper",
    "lutjanus_carponotatus": "Carponotatus Snapper",
    "lutjanus_decussatus": "Decussatus Snapper",
    "lutjanus_ehrenbergii": "Blackspot Snapper",
    "lutjanus_erythropterus": "Erythropterus Snapper",
    "lutjanus_fulviflamma": "Dory Snapper",
    "lutjanus_fulvus": "Blacktail Snapper",
    "lutjanus_gibbus": "Humpback Red Snapper",
    "lutjanus_johnii": "Johnii Snapper",
    "lutjanus_kasmira": "Bluestripe Snapper",
    "lutjanus_lemniscatus": "Lemniscatus Snapper",
    "lutjanus_lutjanus": "Lutjanus Snapper",
    "lutjanus_malabaricus": "Malabaricus Snapper",
    "lutjanus_monostigma": "Monostigma Snapper",
    "lutjanus_quinquelineatus": "Quinquelineatus Snapper",
    "lutjanus_rivulatus": "Blubberlip Snapper",
    "lutjanus_russellii": "Russellii Snapper",
    "lutjanus_sebae": "Emperor Red Snapper",
    "lutjanus_semicinctus": "Semicinctus Snapper",
    "lutjanus_timoriensis": "Timoriensis Snapper",
    "lutjanus_vitta": "Vitta Snapper",
    "macolor_macularis": "Midnight Snapper",
    "macolor_niger": "Black and White Snapper",
    "macropharyngodon_choati": "Choati Leopard Wrasse",
    "macropharyngodon_kuiteri": "Kuiteri Leopard Wrasse",
    "macropharyngodon_meleagris": "Meleagris Leopard Wrasse",
    "macropharyngodon_negrosensis": "Negrosensis Leopard Wrasse",
    "macropharyngodon_ornatus": "Ornatus Leopard Wrasse",
    "megalaspis_cordyla": "Cordyla Torpedo Scad",
    "meuschenia_australis": "Australis Leatherjacket",
    "meuschenia_freycineti": "Freycineti Leatherjacket",
    "meuschenia_galii": "Galii Leatherjacket",
    "meuschenia_hippocrepis": "Hippocrepis Leatherjacket",
    "meuschenia_scaber": "Scaber Leatherjacket",
    "meuschenia_trachylepis": "Trachylepis Leatherjacket",
    "monacanthus_chinensis": "Chinensis Filefish",
    "monotaxis_grandoculis": "Humpnose Big-eye Bream",
    "mugim_cephalus": "Cephalus Mullet",
    "naucrates_ductor": "Ductor Pilotfish",
    "negaprion_acutidens": "Sicklefin Lemon Shark",
    "nemadactylus_douglasii": "Douglasii Morwong",
    "nemipterus_furcosus": "Furcosus Threadfin Bream",
    "nemipterus_hexodon": "Hexodon Threadfin Bream",
    "nemipterus_peronii": "Peronii Threadfin Bream",
    "netuma_thalassina": "Thalassina Catfish",
    "nibea_soldado": "Soldado Croaker",
    "notolabrus_fucicola": "Fucicola Wrasse",
    "notolabrus_gymnogenis": "Gymnogenis Wrasse",
    "notolabrus_tetricus": "Tetricus Wrasse",
    "notorynchus_cepedianus": "Sevengill Shark",
    "novaculichthys_taeniourus": "Rockmover Wrasse",
    "novaculoides_macrolepidotus": "Macrolepidotus Razorfish",
    "oedalechilus_labiosus": "Labiosus Mullet",
    "ophthalmolepis_lineolatus": "Lineolatus Wrasse",
    "otolithes_ruber": "Ruber Croaker",
    "oxycheilinus_bimaculatus": "Bimaculatus Wrasse",
    "oxycheilinus_celebicus": "Celebicus Wrasse",
    "oxycheilinus_digrammus": "Digrammus Wrasse",
    "oxycheilinus_unifasciatus": "Unifasciatus Wrasse",
    "oxymonacanthus_longirostris": "Longnose Filefish",
    "pagrus_auratus": "Australasian Snapper",
    "paracaesio_kusakarii": "Kusakarii Fusilier",
    "paracheilinus_filamentosus": "Filamentosus Flasher Wrasse",
    "paraluteres_prionurus": "Prionurus Filefish",
    "paramonacanthus_choirocephalus": "Choirocephalus Filefish",
    "paraplagusia_bilineata": "Bilineata Tongue Sole",
    "parastromateus_niger": "Black Pomfret",
    "pardachirus_hedleyi": "Hedleyi Sole",
    "pardachirus_pavoninus": "Pavoninus Sole",
    "pentapodus_aureofasciatus": "Aureofasciatus Threadfin Bream",
    "pentapodus_paradiseus": "Paradiseus Threadfin Bream",
    "pentapodus_vitta_quoy": "Vitta Quoy Threadfin Bream",
    "pervagor_alternans": "Alternans Filefish",
    "pervagor_janthinosoma": "Janthinosoma Filefish",
    "pervagor_melanocephalus": "Melanocephalus Filefish",
    "pervagor_nigrolineatus": "Nigrolineatus Filefish",
    "pinjalo_lewisi": "Lewisi Snapper",
    "platycephalus_indicus": "Indicus Flathead",
    "plectranthias_longimanus": "Longimanus Perchlet",
    "plectranthias_nanus": "Nanus Perchlet",
    "plectranthias_winniensis": "Winniensis Perchlet",
    "plectropomus_areolatus": "Squaretail Coral Trout",
    "plectropomus_laevis": "Blacksaddled Coral Trout",
    "plectropomus_leopardus": "Leopard Coral Trout",
    "plectropomus_maculatus": "Spotted Coral Trout",
    "plectropomus_oligacanthus": "Highfin Coral Trout",
    "plotosus_lineatus": "Striped Catfish",
    "pristipomoides_argyrogrammicus": "Argyrogrammicus Snapper",
    "pristipomoides_auricilla": "Auricilla Snapper",
    "pristipomoides_filamentosus": "Filamentosus Snapper",
    "pristipomoides_flavipinnis": "Flavipinnis Snapper",
    "pristipomoides_sieboldii": "Sieboldii Snapper",
    "pristipomoides_zonatus": "Zonatus Snapper",
    "promethichthys_prometheus": "Prometheus Snake Mackerel",
    "protonibea_diacanthus": "Diacanthus Black-spotted Croaker",
    "psettodes_erumei": "Erumei Flounder",
    "pseudalutarius_nasicornis": "Nasicornis Filefish",
    "pseudanthias_bicolor": "Bicolor Anthias",
    "pseudanthias_cooperi": "Cooperi Anthias",
    "pseudanthias_dispar": "Peach Fairy Basslet",
    "pseudanthias_fasciatus": "Fasciatus Anthias",
    "pseudanthias_huchtii": "Red-cheeked Anthias",
    "pseudanthias_hypselosoma": "Hypselosoma Anthias",
    "pseudanthias_lori": "Lori Anthias",
    "pseudanthias_luzonensis": "Luzonensis Anthias",
    "pseudanthias_pictilis": "Pictilis Anthias",
    "pseudanthias_pleurotaenia": "Square-spot Anthias",
    "pseudanthias_rubrizonatus": "Rubrizonatus Anthias",
    "pseudanthias_sheni": "Sheni Anthias",
    "pseudanthias_smithvanizi": "Smithvanizi Anthias",
    "pseudanthias_squamipinnis": "Sea Goldie",
    "pseudanthias_tuka": "Purple Queen",
    "pseudanthias_ventralis": "Ventralis Anthias",
    "pseudocaranx_dentex": "Dentex Trevally",
    "pseudocarcharias_kamoharai": "Crocodile Shark",
    "pseudocheilinus_evanidus": "Evanidus Wrasse",
    "pseudocheilinus_hexataenia": "Sixline Wrasse",
    "pseudocheilinus_ocellatus": "Ocellatus Wrasse",
    "pseudocheilinus_octotaenia": "Octotaenia Wrasse",
    "pseudodax_moluccanus": "Moluccanus Chiseltooth Wrasse",
    "pseudojuloides_cerasinus": "Cerasinus Wrasse",
    "pseudolabrus_biserialis": "Biserialis Wrasse",
    "pseudolabrus_guentheri": "Guentheri Wrasse",
    "pseudolabrus_luculentus": "Luculentus Wrasse",
    "pseudorhombus_argus": "Argus Flounder",
    "pseudorhombus_arsius": "Arsius Flounder",
    "pseudorhombus_elevatus": "Elevatus Flounder",
    "pteragogus_cryptus": "Cryptus Wrasse",
    "pteragogus_enneacanthus": "Enneacanthus Wrasse",
    "pteragogus_flagellifer": "Flagellifer Wrasse",
    "rastrelliger_kanagurta": "Indian Mackerel",
    "retropinna_semoni": "Semoni Smelt",
    "rhabdosargus_sarba": "Sarba Stumpnose",
    "rhincodon_typus": "Whale Shark",
    "rhizoprionodon_acutus": "Milk Shark",
    "ruvettus_pretiosus": "Oilfish",
    "samaris_cristatus": "Cristatus Flounder",
    "samariscus_triocellatus": "Triocellatus Flounder",
    "sarda_orientalis": "Striped Bonito",
    "sardinella_albella": "Albella Sardine",
    "sardinella_gibbosa": "Gibbosa Sardine",
    "sardinops_sagax": "Pacific Sardine",
    "scaevius_milii": "Milii Leatherjacket",
    "scolopsis_affinis": "Affinis Monocle Bream",
    "scolopsis_bilineata": "Two-lined Monocle Bream",
    "scolopsis_lineata": "Lineata Monocle Bream",
    "scolopsis_margaritifer": "Margaritifer Monocle Bream",
    "scolopsis_monogramma": "Monogramma Monocle Bream",
    "scolopsis_trilineata": "Trilineata Monocle Bream",
    "scolopsis_trilineata_4": "Trilineata 4 Monocle Bream",
    "scolopsis_vosmeri": "Vosmeri Monocle Bream",
    "scolopsis_xenochrous": "Xenochrous Monocle Bream",
    "scomberoides_commersonnianus": "Commersonnianus Queenfish",
    "scomberoides_lysan": "Lysan Queenfish",
    "scomberomorus_commerson": "Spanish Mackerel",
    "selar_crumenophthalmus": "Crumenophthalmus Bigeye Scad",
    "selaroides_leptolepis": "Leptolepis Yellowstripe Scad",
    "seriola_dumerili": "Greater Amberjack",
    "seriola_hippos": "Hippos Amberjack",
    "seriola_rivoliana": "Rivoliana Amberjack",
    "seriolina_nigrofasciata": "Nigrofasciata Yellowtail Amberjack",
    "serranocirrhitus_latus": "Latus Hawkfish Anthias",
    "sillago_ciliata": "Ciliata Whiting",
    "sillago_sihama": "Sihama Whiting",
    "soleichthys_heterorhinos": "Heterorhinos Sole",
    "sphyraena_barracuda": "Great Barracuda",
    "sphyraena_forsteri": "Bigeye Barracuda",
    "sphyraena_jello": "Pickhandle Barracuda",
    "sphyraena_obtusata": "Obtuse Barracuda",
    "stegostoma_fasciatum": "Zebra Shark",
    "stethojulis_bandanensis": "Bandanensis Wrasse",
    "stethojulis_interrupta": "Interrupta Wrasse",
    "stethojulis_strigiventer": "Strigiventer Wrasse",
    "stethojulis_trilineata": "Trilineata Wrasse",
    "stolephorus_waitei": "Waitei Anchovy",
    "suezichthys_arquatus": "Arquatus Wrasse",
    "suezichthys_cyanolaemus": "Cyanolaemus Wrasse",
    "suezichthys_gracilis": "Gracilis Wrasse",
    "symphorichthys_spilurus": "Sailfin Snapper",
    "symphorus_nematophorus": "Nematophorus Snapper",
    "thalassohalichoeres_marginatusma_hardwicke": "Marginatusma Hardwicke Wrasse",
    "thalassoma_amblycephalum": "Twotone Wrasse",
    "thalassoma_hardwicke": "Sixbar Wrasse",
    "thalassoma_jansenii": "Jansen's Wrasse",
    "thalassoma_lunare": "Moon Wrasse",
    "thalassoma_lutescens": "Yellow-brown Wrasse",
    "thalassoma_nigrofasciatum": "Blackbarred Wrasse",
    "thalassoma_purpureum": "Surge Wrasse",
    "thalassoma_quinquevittatum": "Fivestripe Wrasse",
    "thalassoma_trilobatum": "Christmas Wrasse",
    "thryssa_baelama": "Baelama Anchovy",
    "thryssa_hamiltonii": "Hamiltonii Anchovy",
    "thunnus_alalunga": "Albacore Tuna",
    "thunnus_albacares": "Yellowfin Tuna",
    "thysanophrys_celebica": "Celebica Flathead",
    "thysanophrys_chiltonae": "Chiltonae Flathead",
    "trachichthys_australis": "Australis Roughy",
    "trachinotus_baillonii": "Small-spotted Dart",
    "trachinotus_blochii": "Snubnose Pompano",
    "trachinotus_botla": "Botla Dart",
    "trachypoma_macracanthus": "Macracanthus Grouper",
    "triaenodon_obesus": "Whitetip Reef Shark",
    "uraspis_secunda": "Secunda Trevally",
    "valamugil_cunnesius": "Cunnesius Mullet",
    "valamugil_engeli": "Engeli Mullet",
    "valamugil_seheli": "Seheli Mullet",
    "variola_albimarginata": "White-edged Lyretail",
    "variola_louti": "Yellow-edged Lyretail",
    "wattsia_mossambica": "Mossambica Western Butterfish",
    "wetmorella_albofasciata": "Albofasciata Possum Wrasse",
    "wetmorella_nigropinnata": "Nigropinnata Possum Wrasse",
    "xiphocheilus_typus": "Typus Tuskfish",
    "zenarchopterus_dispar": "Dispar Halfbeak",
    "zeus_faber": "John Dory",
}


def common_name(model_name: str) -> str:
    if model_name in COMMON_NAMES:
        return COMMON_NAMES[model_name]
    if "_" in model_name:
        return model_name.replace("_", " ").title()
    return model_name


def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_cross_model(detections: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    """Remove duplicate detections of the same fish by different models on the same frame."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for d in detections:
        by_frame[d["frame_index"]].append(d)

    keep = set(range(len(detections)))
    idx_map: dict[int, int] = {}
    pos = 0
    for frame_dets in by_frame.values():
        for d in frame_dets:
            idx_map[id(d)] = pos
            pos += 1

    idx_map = {}
    for i, d in enumerate(detections):
        idx_map[id(d)] = i

    for frame_dets in by_frame.values():
        for a, b in combinations(frame_dets, 2):
            ia, ib = idx_map[id(a)], idx_map[id(b)]
            if ia not in keep or ib not in keep:
                continue
            bbox_a = a.get("bbox")
            bbox_b = b.get("bbox")
            if bbox_a is None or bbox_b is None:
                continue
            if _iou(bbox_a, bbox_b) >= iou_thresh:
                spec_a = a.get("specificity", 0)
                spec_b = b.get("specificity", 0)
                if spec_a != spec_b:
                    loser = ib if spec_a > spec_b else ia
                else:
                    loser = ib if a["confidence"] >= b["confidence"] else ia
                keep.discard(loser)

    return [detections[i] for i in sorted(keep)]


def _merge_fragmented_tracks(
    tracks: dict[str, list[dict]], gap_threshold_sec: float = 15.0
) -> dict[str, list[dict]]:
    """Merge tracks of the same species that never co-occur on the same frame.

    The tracker frequently re-assigns IDs to the same fish — it may switch
    from track 5 to track 10 mid-swim, then back to 5. Checking time-range
    overlap misses this because the ranges interleave. Instead we check
    actual frame-level co-occurrence: two tracks that share the same species
    and never appear on the same frame are the same animal.
    """
    species_groups: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for track_id, dets in tracks.items():
        species = Counter(d["species"] for d in dets).most_common(1)[0][0]
        name = common_name(species)
        species_groups[name].append((track_id, dets))

    merged: dict[str, list[dict]] = {}
    for _species, track_list in species_groups.items():
        track_list.sort(key=lambda t: min(d["timestamp_sec"] for d in t[1]))

        clusters: list[tuple[str, list[dict], set[int]]] = []
        for track_id, dets in track_list:
            frames = {d["frame_index"] for d in dets}
            placed = False
            for ci, (cid, cdets, cframes) in enumerate(clusters):
                if not frames & cframes:
                    last_cur = max(d["timestamp_sec"] for d in cdets)
                    first_new = min(d["timestamp_sec"] for d in dets)
                    last_new = max(d["timestamp_sec"] for d in dets)
                    gap = max(0, first_new - last_cur)
                    if gap <= gap_threshold_sec:
                        clusters[ci] = (cid, cdets + dets, cframes | frames)
                        placed = True
                        break
            if not placed:
                clusters.append((track_id, dets, frames))

        for cid, cdets, _ in clusters:
            merged[cid] = cdets

    return merged


def _dedup_tracks_spatially(
    tracks: dict[str, list[dict]], iou_thresh: float = 0.3
) -> dict[str, list[dict]]:
    """Merge tracks whose bounding boxes overlap across frames.

    Both models often detect the same physical fish but assign different
    species labels (e.g. FishInv calls an Emperor Angelfish "serranidae"
    while Seychelles correctly identifies it).  Per-frame IoU dedup catches
    same-frame overlaps, but the models don't always fire on the same
    frames.  This pass checks bbox overlap on shared *and nearby* frames
    (±2) and merges tracks that clearly follow the same animal.
    """
    track_ids = list(tracks.keys())
    if len(track_ids) <= 1:
        return tracks

    frame_map = {tid: {d["frame_index"]: d for d in dets} for tid, dets in tracks.items()}

    parent = {tid: tid for tid in track_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if len(tracks[ra]) < len(tracks[rb]):
                ra, rb = rb, ra
            parent[rb] = ra

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            tid_a, tid_b = track_ids[i], track_ids[j]
            fa, fb = frame_map[tid_a], frame_map[tid_b]

            overlaps = 0
            comparisons = 0
            for fidx, det_a in fa.items():
                bbox_a = det_a.get("bbox")
                if not bbox_a:
                    continue
                for offset in range(-2, 3):
                    neighbor = fidx + offset
                    if neighbor in fb:
                        bbox_b = fb[neighbor].get("bbox")
                        if bbox_b:
                            comparisons += 1
                            if _iou(bbox_a, bbox_b) >= iou_thresh:
                                overlaps += 1

            if comparisons >= 2 and overlaps / comparisons >= 0.5:
                union(tid_a, tid_b)

    groups: dict[str, list[dict]] = defaultdict(list)
    for tid in track_ids:
        groups[find(tid)].extend(tracks[tid])
    return dict(groups)


MIN_ACCEPTED_DETECTIONS = 5
SCHOOL_THRESHOLD = 5

# Nothing below this confidence is written to the report as an identification --
# weaker tracks land in the review queue instead, regardless of the per-model
# accept threshold (which can be as low as 0.75).
MIN_WRITE_CONF = 0.85


def _detect_schools(tracks: dict[str, list[dict]], track_majority: dict[str, str]) -> set[str]:
    """Species with SCHOOL_THRESHOLD+ simultaneous tracks on any single frame.

    Tracks are keyed by their majority species -- the same label
    aggregate_tracks routes with -- so a species can never be flagged as a
    school that no track actually resolves to (which would make the school
    silently disappear from the report).
    """
    frame_species: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for track_id, dets in tracks.items():
        species = track_majority[track_id]
        for d in dets:
            frame_species[d["frame_index"]][species].add(track_id)

    schools = set()
    for species_tracks in frame_species.values():
        for species, tids in species_tracks.items():
            if len(tids) >= SCHOOL_THRESHOLD:
                schools.add(species)
    return schools


def aggregate_tracks(detections: list[dict], rarity_map: dict[str, str]) -> dict:
    detections = _dedup_cross_model(detections)

    raw_tracks: dict[str, list[dict]] = defaultdict(list)
    for detection in detections:
        raw_tracks[detection["track_id"]].append(detection)

    tracks = _dedup_tracks_spatially(raw_tracks)
    tracks = _merge_fragmented_tracks(tracks)

    # One majority label per track, used consistently for school detection and
    # school routing below so the two can never disagree.
    track_majority = {
        tid: Counter(d["species"] for d in dets).most_common(1)[0][0]
        for tid, dets in tracks.items()
    }
    school_species = _detect_schools(tracks, track_majority)

    species_counts: Counter = Counter()
    species_conf_sum: dict[str, float] = defaultdict(float)
    species_conf_n: dict[str, int] = defaultdict(int)
    review_queue: list[dict] = []
    accepted_tracks: list[tuple[str, str, list[dict]]] = []
    # A schooling species is reported as one school, not per fish -- pool every
    # member track's detections and summarise them once below.
    school_groups: dict[str, list[dict]] = defaultdict(list)

    for track_id, track_detections in tracks.items():
        track_species = track_majority[track_id]
        if track_species in school_species:
            school_groups[track_species].extend(track_detections)
            continue

        accept_conf = max(track_detections[0]["accept_conf"], MIN_WRITE_CONF)
        accepted = [d for d in track_detections if d["confidence"] >= accept_conf]

        if len(accepted) >= MIN_ACCEPTED_DETECTIONS:
            species = Counter(d["species"] for d in accepted).most_common(1)[0][0]
            avg_conf = sum(d["confidence"] for d in accepted) / len(accepted)
            species_counts[species] += 1
            species_conf_sum[species] += avg_conf
            species_conf_n[species] += 1
            accepted_tracks.append((track_id, species, accepted))
        else:
            best = max(track_detections, key=lambda d: d["confidence"])
            review_queue.append({
                "track_id": track_id,
                "species_guess": best["species"],
                "max_confidence": round(best["confidence"], 3),
                "timestamp_sec": best["timestamp_sec"],
                "crop_path": best["crop_path"],
            })

    species_summary = []
    for species, count in species_counts.most_common():
        species_summary.append({
            "species": common_name(species),
            "unique_count": count,
            "avg_confidence": round(species_conf_sum[species] / species_conf_n[species], 3),
            "rarity": rarity_map.get(species, "unknown"),
            "is_school": False,
        })

    # Individual sightings are built from the accepted (>= MIN_WRITE_CONF)
    # detections only, and reuse the exact species label the summary counted --
    # sub-threshold detections are never written to the report.
    track_details = []
    for track_id, species, accepted in accepted_tracks:
        best = max(accepted, key=lambda d: d["confidence"])
        timestamps = sorted({d["timestamp_sec"] for d in accepted})
        track_details.append({
            "track_id": track_id,
            "species": common_name(species),
            "detection_count": len(accepted),
            "best_confidence": round(best["confidence"], 3),
            "best_bbox": best.get("bbox"),
            "best_crop": best.get("crop_path"),
            "first_seen": min(timestamps),
            "last_seen": max(timestamps),
            "timestamps": timestamps,
        })

    for species, dets in school_groups.items():
        # Prefer detections that both clear the floor and carry the school's
        # own label; fall back to any confident detection from member tracks.
        confident = [d for d in dets if d["confidence"] >= MIN_WRITE_CONF and d["species"] == species]
        if not confident:
            confident = [d for d in dets if d["confidence"] >= MIN_WRITE_CONF]
        if not confident:
            best = max(dets, key=lambda d: d["confidence"])
            review_queue.append({
                "track_id": f"school_{species}",
                "species_guess": f"School of {common_name(species)}",
                "max_confidence": round(best["confidence"], 3),
                "timestamp_sec": best["timestamp_sec"],
                "crop_path": best["crop_path"],
            })
            continue
        best = max(confident, key=lambda d: d["confidence"])
        avg_conf = sum(d["confidence"] for d in confident) / len(confident)
        species_summary.append({
            "species": f"School of {common_name(species)}",
            "unique_count": 1,
            "avg_confidence": round(avg_conf, 3),
            "rarity": rarity_map.get(species, "unknown"),
            "is_school": True,
        })
        timestamps = sorted({d["timestamp_sec"] for d in confident})
        track_details.append({
            "track_id": f"school_{species}",
            "species": f"School of {common_name(species)}",
            "detection_count": len(confident),
            "best_confidence": round(best["confidence"], 3),
            "best_bbox": best.get("bbox"),
            "best_crop": best.get("crop_path"),
            "first_seen": min(timestamps),
            "last_seen": max(timestamps),
            "timestamps": timestamps,
        })

    for item in review_queue:
        item["species_guess"] = common_name(item["species_guess"])

    review_queue.sort(key=lambda r: r["timestamp_sec"])
    track_details.sort(key=lambda t: t["first_seen"])

    return {
        "species_summary": species_summary,
        "review_queue": review_queue,
        "track_details": track_details,
    }
