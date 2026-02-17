import networkx as nx
from pyvis.network import Network
import numpy as np
import pandas as pd

# Extended MCC mapping for common payment categories
extended_mcc_mapping = {
    '0': 'Unclassified',
    '2741': 'Miscellaneous Publishing and Printing',
    '4111': 'Transportation - Suburban and Local Commuter Passenger, including Ferries',  
    '4121': 'Taxicabs and Limousines',
    '4215': 'Courier Services - Air and Ground, Freight Forwarders',
    '4225': 'Public Warehousing, Storage',
    '4722': 'Travel Agencies and Tour Operators',
    '4789': 'Transportation Services',
    '4812': 'Telecommunications Equipment including telephone sales',
    '4814': 'Telecommunication Services',
    '4816': 'Computer Network/Information Services',
    '4829': 'Wire Transfers/Money Orders',
    '4899': 'Cable and Other Pay Television',
    '4900': 'Electric, Gas, Sanitary and Water Utilities',
    '5039': 'Construction Materials - Not Elsewhere Classified',
    '5045': 'Computers, Computer Peripheral Equipment, Software',
    '5065': 'Electrical Parts and Equipment',
    '5074': 'Plumbing and Heating Equipment',
    '5085': 'Industrial Supplies',
    '5199': 'Nondurable Goods',
    '5200': 'Home Supply Warehouse Stores',
    '5211': 'Lumber and Building Materials Stores',
    '5231': 'Glass, Paint, and Wallpaper Stores',
    '5251': 'Hardware Stores',
    '5261': 'Lawn and Garden Supply Stores',
    '5300': 'Wholesale Clubs',
    '5310': 'Discount Stores',
    '5311': 'Department Stores',
    '5331': 'Variety Stores',
    '5411': 'Grocery Stores/Supermarkets',
    '5422': 'Freezer and Locker Meat Provisioners',
    '5441': 'Candy, Nut, and Confectionery Stores',
    '5451': 'Dairy Products Stores',
    '5462': 'Bakeries',
    '5499': 'Miscellaneous Food Stores',
    '5511': 'Car and Truck Dealers (New and Used)',
    '5521': 'Car and Truck Dealers (Used Only)',
    '5531': 'Auto and Home Supply Stores',
    '5532': 'Automotive Tire Stores',
    '5533': 'Automotive Parts and Accessories Stores',
    '5541': 'Service Stations',
    '5542': 'Automated Fuel Dispensers',
    '5551': 'Boat Dealers',
    '5561': 'Camper, Recreational and Utility Trailer Dealers',
    '5571': 'Motorcycle Shops and Dealers',
    '5592': 'Motor Home Dealers',
    '5598': 'Snowmobile Dealers',
    '5599': 'Miscellaneous Automotive, Aircraft, and Farm Equipment Dealers',
    '5611': "Men's and Boy's Clothing and Accessories Stores",
    '5621': "Women's Ready-to-Wear Stores",
    '5631': "Women's Accessory and Specialty Shops",
    '5641': "Children's and Infant's Wear Stores",
    '5651': 'Family Clothing Stores',
    '5655': 'Sports and Riding Apparel Stores',
    '5661': 'Shoe Stores',
    '5681': 'Furriers and Fur Shops',
    '5691': "Men's and Women's Clothing Stores",
    '5697': 'Tailors, Seamstresses, Mending, and Alterations',
    '5698': 'Wig and Toupee Stores',
    '5699': 'Miscellaneous Apparel and Accessory Shops',
    '5712': 'Furniture, Home Furnishings, and Equipment Stores',
    '5713': 'Floor Covering Stores',
    '5714': 'Drapery, Window Covering and Upholstery Stores',
    '5718': 'Fireplace, Fireplace Screens, and Accessories Stores',
    '5719': 'Miscellaneous Home Furnishing Specialty Stores',
    '5722': 'Household Appliance Stores',
    '5732': 'Electronics Stores',
    '5733': 'Music Stores',
    '5734': 'Computer Software Stores',
    '5735': 'Record Stores',
    '5811': 'Caterers',
    '5812': 'Eating Places and Restaurants',
    '5813': 'Drinking Places (Alcoholic Beverages)',
    '5814': 'Fast Food Restaurants',
    '5815': 'Digital Goods: Media - Books, Movies, Music',
    '5816': 'Digital Goods: Games',
    '5817': 'Digital Goods: Applications',
    '5818': 'Digital Goods: Large Digital Goods Merchant',
    '5912': 'Drug Stores and Pharmacies',
    '5921': 'Package Stores - Beer, Wine, and Liquor',
    '5931': 'Used Merchandise and Secondhand Stores',
    '5932': 'Antique Shops - Sales, Repairs, and Restoration Services',
    '5933': 'Pawn Shops',
    '5935': 'Wrecking and Salvage Yards',
    '5937': 'Antique Reproduction Stores',
    '5940': 'Bicycle Shops - Sales and Service',
    '5941': 'Sporting Goods Stores',
    '5942': 'Book Stores',
    '5943': 'Stationery, Office and School Supply Stores',
    '5944': 'Jewelry, Watch, Clock, and Silverware Stores',
    '5945': 'Hobby, Toy, and Game Shops',
    '5946': 'Camera and Photographic Supply Stores',
    '5947': 'Card Shops, Gift, Novelty, and Souvenir Shops',
    '5948': 'Leather Goods and Luggage Stores',
    '5949': 'Sewing, Needlework, Fabric, and Piece Goods Stores',
    '5950': 'Glassware/Crystal Stores',
    '5960': 'Direct Marketing - Insurance Services',
    '5962': 'Direct Marketing - Travel Related Arrangements Services',
    '5963': 'Door-to-Door Sales',
    '5964': 'Direct Marketing - Catalog Merchant',
    '5965': 'Direct Marketing - Combination Catalog and Retail Merchant',
    '5966': 'Direct Marketing - Outbound Telemarketing Merchant',
    '5967': 'Direct Marketing - Inbound Teleservices Merchant',
    '5968': 'Direct Marketing - Continuity/Subscription Merchant',
    '5969': 'Direct Marketing - Not Elsewhere Classified',
    '5970': 'Artist Supply and Craft Shops',
    '5971': 'Art Dealers and Galleries',
    '5972': 'Stamp and Coin Stores',
    '5973': 'Religious Goods Stores',
    '5975': 'Hearing Aids - Sales, Service, and Supply Stores',
    '5976': 'Orthopedic Goods - Prosthetic Devices',
    '5977': 'Cosmetic Stores',
    '5978': 'Typewriter Stores',
    '5983': 'Fuel Dealers',
    '5992': 'Florists',
    '5993': 'Cigar Stores and Stands',
    '5994': 'News Dealers and Newsstands',
    '5995': 'Pet Shops, Pet Food, and Supplies',
    '5996': 'Swimming Pools - Sales, Service, and Supplies',
    '5997': 'Electric Razor Stores',
    '5998': 'Tent and Awning Shops',
    '5999': 'Miscellaneous and Specialty Retail Stores',
    '6010': 'Financial Institutions - Manual Cash Disbursements',
    '6011': 'Financial Institutions - Automated Cash Disbursements',
    '6012': 'Financial Institutions - Merchandise and Services',
    '6051': 'Non-Financial Institutions - Foreign Currency, Money Orders',
    '6211': 'Security Brokers/Dealers',
    '6300': 'Insurance Sales, Underwriting, and Premiums',
    '6513': 'Real Estate Agents and Managers - Rentals',
    '7011': 'Lodging - Hotels, Motels, Resorts',
    '7012': 'Timeshares',
    '7032': 'Sporting and Recreational Camps',
    '7033': 'Trailer Parks and Camp Sites',
    '7210': 'Laundry, Cleaning, and Garment Services',
    '7211': 'Laundries - Family and Commercial',
    '7216': 'Dry Cleaners',
    '7217': 'Carpet and Upholstery Cleaning',
    '7221': 'Photographic Studios',
    '7230': 'Barber and Beauty Shops',
    '7251': 'Shoe Repair Shops, Shoe Shine Parlors, Hat Cleaning Shops',
    '7261': 'Funeral Services and Crematories',
    '7273': 'Dating and Escort Services',
    '7276': 'Tax Preparation Services',
    '7277': 'Counseling Services',
    '7278': 'Buying/Shopping Services, Clubs',
    '7296': 'Clothing Rental',
    '7297': 'Massage Parlors',
    '7298': 'Health and Beauty Spas',
    '7299': 'Miscellaneous Personal Services',
    '7311': 'Advertising Services',
    '7321': 'Consumer Credit Reporting Agencies',
    '7333': 'Commercial Photography, Art and Graphics',
    '7338': 'Quick Copy, Reproduction, and Blueprinting Services',
    '7339': 'Stenographic and Secretarial Support Services',
    '7342': 'Exterminating and Disinfecting Services',
    '7349': 'Cleaning, Maintenance, and Janitorial Services',
    '7361': 'Employment Agencies, Temporary Help Services',
    '7372': 'Computer Programming, Data Processing',
    '7375': 'Information Retrieval Services',
    '7379': 'Computer Maintenance and Repair Services',
    '7392': 'Management, Consulting, and Public Relations Services',
    '7393': 'Detective Agencies, Protective Agencies, Security Services',
    '7394': 'Equipment Rental and Leasing Services',
    '7395': 'Photofinishing Laboratories, Photo Developing',
    '7399': 'Business Services - Not Elsewhere Classified',
    '7512': 'Automobile Rental Agency',
    '7513': 'Truck and Utility Trailer Rentals',
    '7519': 'Motor Home and Recreational Vehicle Rentals',
    '7523': 'Parking Lots, Parking Meters, Garages',
    '7531': 'Automotive Body Repair Shops',
    '7534': 'Tire Re-treading and Repair Shops',
    '7535': 'Paint Shops - Automotive',
    '7538': 'Automotive Service Shops',
    '7542': 'Car Washes',
    '7549': 'Towing Services',
    '7622': 'Electronics Repair Shops',
    '7623': 'Air Conditioning and Refrigeration Repair Shops',
    '7629': 'Electrical and Small Appliance Repair Shops',
    '7631': 'Watch, Clock, and Jewelry Repair',
    '7641': 'Furniture - Reupholstery, Repair, and Refinishing',
    '7692': 'Welding Services',
    '7699': 'Miscellaneous Repair Shops and Related Services',
    '7829': 'Motion Picture and Video Tape Production and Distribution',
    '7832': 'Motion Picture Theaters',
    '7841': 'Video Tape Rental Stores',
    '7911': 'Dance Halls, Studios, and Schools',
    '7922': 'Theatrical Producers (except Motion Pictures), Ticket Agencies',
    '7929': 'Bands, Orchestras, and Miscellaneous Entertainers',
    '7932': 'Billiard and Pool Establishments',
    '7933': 'Bowling Alleys',
    '7941': 'Commercial Sports, Professional Sports Clubs, Athletic Fields',
    '7991': 'Tourist Attractions and Exhibits',
    '7992': 'Public Golf Courses',
    '7993': 'Video Amusement Game Supplies',
    '7994': 'Video Game Arcades/Establishments',
    '7995': 'Betting (including Lottery Tickets, Casino Gaming Chips, Off-track Betting)',
    '7996': 'Amusement Parks, Circuses, Carnivals, and Fortune Tellers',
    '7997': 'Membership Clubs (Sports, Recreation, Athletic), Country Clubs',
    '7998': 'Aquariums, Seaquariums, Dolphinariums',
    '7999': 'Recreation Services - Not Elsewhere Classified',
    '8011': 'Doctors and Physicians',
    '8021': 'Dentists and Orthodontists',
    '8031': 'Osteopaths',
    '8041': 'Chiropractors',
    '8042': 'Optometrists and Ophthalmologists',
    '8043': 'Opticians, Optical Goods, and Eyeglasses',
    '8049': 'Podiatrists and Chiropodists',
    '8050': 'Nursing and Personal Care Facilities',
    '8062': 'Hospitals',
    '8071': 'Medical and Dental Laboratories',
    '8099': 'Medical Services and Health Practitioners',
    '8111': 'Legal Services and Attorneys',
    '8211': 'Elementary and Secondary Schools',
    '8220': 'Colleges, Universities, Professional Schools, and Junior Colleges',
    '8241': 'Correspondence Schools',
    '8244': 'Business and Secretarial Schools',
    '8249': 'Vocational Schools and Trade Schools',
    '8299': 'Schools and Educational Services',
    '8351': 'Child Care Services',
    '8398': 'Charitable and Social Service Organizations',
    '8641': 'Civic, Social, and Fraternal Associations',
    '8651': 'Political Organizations',
    '8661': 'Religious Organizations',
    '8675': 'Automobile Associations',
    '8699': 'Membership Organizations',
    '8734': 'Testing Laboratories (non-medical)',
    '8911': 'Architectural, Engineering, and Surveying Services',
    '8931': 'Accounting, Auditing, and Bookkeeping Services',
    '8999': 'Professional Services',
    '9211': 'Court Costs, including Alimony and Child Support',
    '9222': 'Fines',
    '9223': 'Bail and Bond Payments',
    '9311': 'Tax Payments',
    '9399': 'Government Services',
    '9402': 'Postal Services - Government Only',
    '9405': 'Intra-Government Purchases - Government Only',
    '9700': 'Automated Referral Service',
    '9701': 'Visa Credential Service',
    '9702': 'GCAS Emergency Services',
    '9950': 'Intra-Company Purchases'
}

# Apply comprehensive mapping: kyc_industry_codes first, then MCCs
def map_category(cat_str):
    industry_codes = pd.read_csv("data/kyc_industry_codes.csv.gz")
    industry_map_int = dict(zip(industry_codes['industry_code'], industry_codes['industry']))
    # Try as integer for kyc_industry_codes
    try:
        cat_int = int(cat_str)
        if cat_int in industry_map_int:
            return industry_map_int[cat_int]
    except (ValueError, TypeError):
        pass
    
    # Try MCC mapping
    if cat_str in extended_mcc_mapping:
        return extended_mcc_mapping[cat_str]
    
    # Keep original if not numeric
    if not cat_str.isdigit():
        return cat_str
    
    # Fallback for unmapped numbers
    return f'Industry Code {cat_str}'

def visualize_customer_neighborhood(customer_id, master_pool, customer_pool, data, k_hop=2):
    """
    Visualize customer transaction neighborhood with detailed node information.
    Click on any node to see full details.
    """
    # 1. Initialize NetworkX graph
    G = nx.Graph()
    
    # 2. Find the index for the target customer
    cust_idx = master_pool[master_pool['customer_id'] == customer_id]['cust_idx'].iloc[0]
    
    # 3. Get customer information
    cust_info = customer_pool[customer_pool['customer_id'] == customer_id].iloc[0]
    
    # 4. Filter transactions for this customer
    relevant_txs = master_pool[master_pool['customer_id'] == customer_id]
    
    # Extract customer info as Python native types to avoid numpy array formatting issues
    age = float(cust_info['age'])
    income = float(cust_info['income'])
    tenure = float(cust_info['tenure'])
    is_biz = bool(cust_info['is_biz'])
    sales = float(cust_info['sales'])
    emp_count = float(cust_info['emp_count'])
    
    # Extract transaction aggregations as Python native floats
    avg_amount = float(relevant_txs['amount_cad'].mean())
    max_fraud_risk = float(relevant_txs['fraud_probability'].max())
    
    # Add Central Customer Node with detailed information (single-line HTML for PyVis)
    customer_tooltip = (
        f"<b>CUSTOMER PROFILE</b><br>"
        f"{'━' * 30}<br>"
        f"ID: {customer_id}<br>"
        f"Age: {age:.0f} years<br>"
        f"Income: ${income:,.2f}<br>"
        f"Tenure: {tenure:.0f} days<br>"
        f"Type: {'Business' if is_biz else 'Individual'}<br>"
        f"{'Sales: $' + f'{sales:,.2f}' + '<br>' if is_biz else ''}"
        f"{'Employees: ' + f'{emp_count:.0f}' + '<br>' if is_biz else ''}"
        f"{'━' * 30}<br>"
        f"Total Transactions: {len(relevant_txs)}<br>"
        f"Avg Amount: ${avg_amount:.2f}<br>"
        f"Max Fraud Risk: {max_fraud_risk:.2%}"
    )
    
    G.add_node(customer_id, 
               label=f"Customer\n{customer_id[:12]}...", 
               color='#00FF00', 
               size=30,
               title=customer_tooltip)
    
    # Track unique hubs for aggregated stats
    category_stats = {}
    city_stats = {}
    
    # Add Transaction Nodes with full details
    for _, row in relevant_txs.head(20).iterrows(): # Limit to 20 txs for clarity
        tx_id = f"TX_{row['transaction_id']}"
        risk = float(row['fraud_probability'])
        
        # Color based on risk (Green to Red)
        color = f'rgb({int(255*risk)}, {int(255*(1-risk))}, 0)'
        
        # Transaction tooltip with all available information (single-line HTML for PyVis)
        tx_tooltip = (
            f"<b>TRANSACTION DETAILS</b><br>"
            f"{'━' * 30}<br>"
            f"ID: {row['transaction_id']}<br>"
            f"Amount: ${float(row['amount_cad']):,.2f} CAD<br>"
            f"Type: {row['debit_credit'].upper()}<br>"
            f"Date: {row['transaction_datetime']}<br>"
            f"{'━' * 30}<br>"
            f"Merchant: {row['merchant_category']}<br>"
            f"Location: {row['city']}, {row['province']}<br>"
            f"Country: {row['country']}<br>"
            f"{'━' * 30}<br>"
            f"E-commerce: {'Yes' if row['ecommerce_ind'] else 'No'}<br>"
            f"Cash: {'Yes' if row['cash_indicator'] else 'No'}<br>"
            f"Source: {row['source_dataset']}<br>"
            f"{'━' * 30}<br>"
            f"Time Since Last: {float(row['time_delta']):.0f} min<br>"
            f"24h Velocity: {float(row['velocity_24h']):.0f} txns<br>"
            f"{'━' * 30}<br>"
            f"<b style='color: {'red' if risk > 0.5 else 'green'}'>"
            f"FRAUD RISK: {risk:.1%}</b><br>"
            f"Actual Label: {'FRAUD' if row['is_fraud'] == 1 else 'LEGIT' if row['is_fraud'] == 0 else 'UNKNOWN'}"
        )
        
        G.add_node(tx_id, 
                   label=f"${float(row['amount_cad']):.0f}\n{risk:.0%}", 
                   color=color, 
                   size=15,
                   title=tx_tooltip)
        G.add_edge(customer_id, tx_id)
        
        # Track category stats
        cat = row['merchant_category']
        if cat not in category_stats:
            category_stats[cat] = {'count': 0, 'total_amount': 0, 'avg_risk': []}
        category_stats[cat]['count'] += 1
        category_stats[cat]['total_amount'] += float(row['amount_cad'])
        category_stats[cat]['avg_risk'].append(risk)
        
        # Track city stats
        city = row['city']
        if city not in city_stats:
            city_stats[city] = {'count': 0, 'total_amount': 0, 'avg_risk': []}
        city_stats[city]['count'] += 1
        city_stats[city]['total_amount'] += float(row['amount_cad'])
        city_stats[city]['avg_risk'].append(risk)
        
        # Add Merchant Category Hub
        if not G.has_node(cat):
            cat_tooltip = (
                f"<b>MERCHANT CATEGORY</b><br>"
                f"{'━' * 30}<br>"
                f"{cat}<br>"
                f"{'━' * 30}<br>"
                f"Transactions: {category_stats[cat]['count']}<br>"
                f"Total Amount: ${category_stats[cat]['total_amount']:,.2f}<br>"
                f"Avg Fraud Risk: {np.mean(category_stats[cat]['avg_risk']):.1%}"
            )
            G.add_node(cat, 
                       label=cat[:20] + '...' if len(cat) > 20 else cat,
                       color='#0000FF', 
                       size=20,
                       title=cat_tooltip)
        G.add_edge(tx_id, cat)
        
        # Add City Hub
        if not G.has_node(city):
            city_tooltip = (
                f"<b>LOCATION</b><br>"
                f"{'━' * 30}<br>"
                f"City: {city}<br>"
                f"Province: {row['province']}<br>"
                f"Country: {row['country']}<br>"
                f"{'━' * 30}<br>"
                f"Transactions: {city_stats[city]['count']}<br>"
                f"Total Amount: ${city_stats[city]['total_amount']:,.2f}<br>"
                f"Avg Fraud Risk: {np.mean(city_stats[city]['avg_risk']):.1%}"
            )
            G.add_node(city, 
                       label=city[:15] + '...' if len(city) > 15 else city,
                       color='#FFA500', 
                       size=20,
                       title=city_tooltip)
        G.add_edge(tx_id, city)

    # 4. Export to PyVis with inline resources (embeds everything in HTML for portability)
    net = Network(
        height='750px', 
        width='100%', 
        bgcolor='#222222', 
        font_color='white', # type: ignore
        notebook=False,  # Don't try to display in notebook
        cdn_resources='in_line'  # Embed all JS/CSS in HTML file for portability
    )
    net.from_nx(G)
    
    # Configure physics and interaction (without show_buttons to avoid conflicts)
    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -30000,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 100
        }
      }
    }
    """)
    
    # Save to self-contained HTML file
    filename = f"fraud_map_{customer_id}.html"
    net.save_graph(filename)
    
    print(f"✓ Self-contained HTML saved to {filename}")
    print(f"✓ Graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"✓ File is ready to copy to another computer (no internet needed)")
    
    return filename