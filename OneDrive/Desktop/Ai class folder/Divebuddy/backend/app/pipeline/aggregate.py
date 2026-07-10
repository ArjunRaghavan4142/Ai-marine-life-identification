"""Turns raw per-frame detections into a biodiversity summary.

Two things happen here that the raw detection list can't give you on its own:
  - duplicate-count reduction: a track (one animal followed across frames)
    counts once per species, not once per frame it appeared in
  - low-confidence review: a track that never crossed its model's accept
    threshold in any frame goes into a review queue instead of being counted
"""
from collections import Counter, defaultdict

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
}


def common_name(model_name: str) -> str:
    return COMMON_NAMES.get(model_name, model_name)


def _merge_fragmented_tracks(
    tracks: dict[str, list[dict]], gap_threshold_sec: float = 15.0
) -> dict[str, list[dict]]:
    """Merge tracks of the same species that are sequential and close in time.

    The tracker loses a fish when the camera pans away or the fish briefly
    leaves frame, then assigns a new track ID when it reappears.  If two
    tracks share the same majority species, don't overlap in time, and the
    gap between them is <= gap_threshold_sec, they're almost certainly the
    same animal — merge them under one track ID.  Tracks that overlap in
    time are genuinely different animals and are left alone.
    """
    species_groups: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for track_id, dets in tracks.items():
        species = Counter(d["species"] for d in dets).most_common(1)[0][0]
        species_groups[species].append((track_id, dets))

    merged: dict[str, list[dict]] = {}
    for _species, track_list in species_groups.items():
        track_list.sort(key=lambda t: min(d["timestamp_sec"] for d in t[1]))

        cur_id, cur_dets = track_list[0]
        cur_end = max(d["timestamp_sec"] for d in cur_dets)

        for track_id, dets in track_list[1:]:
            t_start = min(d["timestamp_sec"] for d in dets)
            t_end = max(d["timestamp_sec"] for d in dets)

            if t_start > cur_end and (t_start - cur_end) <= gap_threshold_sec:
                cur_dets = cur_dets + dets
                cur_end = max(cur_end, t_end)
            else:
                merged[cur_id] = cur_dets
                cur_id, cur_dets = track_id, dets
                cur_end = t_end

        merged[cur_id] = cur_dets

    return merged


def aggregate_tracks(detections: list[dict], rarity_map: dict[str, str]) -> dict:
    raw_tracks: dict[str, list[dict]] = defaultdict(list)
    for detection in detections:
        raw_tracks[detection["track_id"]].append(detection)

    tracks = _merge_fragmented_tracks(raw_tracks)

    species_counts: Counter = Counter()
    species_conf_sum: dict[str, float] = defaultdict(float)
    species_conf_n: dict[str, int] = defaultdict(int)
    review_queue: list[dict] = []

    for track_id, track_detections in tracks.items():
        accept_conf = track_detections[0]["accept_conf"]
        accepted = [d for d in track_detections if d["confidence"] >= accept_conf]

        if accepted:
            species = Counter(d["species"] for d in accepted).most_common(1)[0][0]
            avg_conf = sum(d["confidence"] for d in accepted) / len(accepted)
            species_counts[species] += 1
            species_conf_sum[species] += avg_conf
            species_conf_n[species] += 1
        else:
            best = max(track_detections, key=lambda d: d["confidence"])
            review_queue.append({
                "track_id": track_id,
                "species_guess": best["species"],
                "max_confidence": round(best["confidence"], 3),
                "timestamp_sec": best["timestamp_sec"],
                "crop_path": best["crop_path"],
            })

    species_summary = [
        {
            "species": common_name(species),
            "unique_count": count,
            "avg_confidence": round(species_conf_sum[species] / species_conf_n[species], 3),
            "rarity": rarity_map.get(species, "unknown"),
        }
        for species, count in species_counts.most_common()
    ]

    for item in review_queue:
        item["species_guess"] = common_name(item["species_guess"])

    review_queue.sort(key=lambda r: r["timestamp_sec"])
    return {"species_summary": species_summary, "review_queue": review_queue}
