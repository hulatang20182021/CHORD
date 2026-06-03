# Beauty CF-SVD Fallback Embedding

## Why Not SASRec item_feature_matrix_cf.npy

The prior SASRec audit found that the existing SASRec CF feature matrix has plausible shape, but its item content mapping to LETTER Beauty could not be proven. This build does not use that file.

## Method

- Source: LETTER `data/Beauty/Beauty.inter.json`
- Co-occurrence window: 5
- Pair weight: `1 / distance`, symmetric
- Matrix transform: sparse PPMI
- Embedding: TruncatedSVD with dim 128, then row L2 normalization
- CF cluster: KMeans with 256 clusters

## Alignment Check

- item_id_order aligned with Beauty.index.json: True
- num_items: 12101
- skipped_items: 0

## Matrix And SVD Stats

- cooc_nnz: 1050000
- ppmi_nnz: 1043112
- SVD explained variance sum: 0.11419887095689774

## Norm And Zero Row Check

- row norm mean/median/min/max: 1.0 / 1.0 / 0.9999997615814209 / 1.0000001192092896
- zero_row_count: 0

## CF Cluster Stats

- empty clusters: 0
- min/median/max cluster size: 7 / 46.0 / 134
- neighbor sharing lift: 48.858930602957905

## Alignment With Existing Signals

- NMI(CF cluster, original c1): 0.4237630636292303
- purity(CF cluster, original c1): 0.3269151309809107
- NMI(CF cluster, product_type): 0.2631732516665439
- purity(CF cluster, product_type): 0.3131146186265598

## Recommendation

- recommended_for_cr_letter_l_cf: True
- valid: True
- warnings: []

## Limitations

- This is a CF fallback reconstructed from LETTER Beauty interactions.
- It is not a SASRec checkpoint embedding.
- Its strength is strict LETTER item alignment, making it suitable as a first collaborative regularization input.

## Nearest-Neighbor Examples

### High Exposure
- Query 300 exposure=431 title=Dotting 5 X 2 Way Marbleizing Dotting Pen Set for Nail Art Manicure Pedicure, 4 Ounce
  - 278 cos=0.9401 exposure=391 title=World Pride Nail Tape Stripe Decoration Sticker Hologram, Set of 10
  - 301 cos=0.9279 exposure=321 title=15pcs Nail Art Painting Pen Brush
  - 277 cos=0.9253 exposure=282 title=MASH Rhinestones 2400 Piece 12 Color Nail Art Nailart Manicure Wheels
  - 2009 cos=0.9056 exposure=151 title=5pcs 2-ways Acrylic Uv Gel Nail Art Design Tips Dotting Painting Brush Pen Set
  - 7842 cos=0.9003 exposure=79 title=Set of 3 Sable NAIL ART Brushes Pen, Detailer Liner and Striper
  - 295 cos=0.8983 exposure=305 title=Nail Art Brushes- Professional Nail Art Brushes- Sable Nail Art Brush Pen, Detailer, Liner **Set of 3
  - 294 cos=0.8962 exposure=113 title=BONAMART &reg; Premium MASH 100 Pc Nail Art Nailart 3d Manicure Design Sticks Rods Stickers Gel Tips
  - 292 cos=0.8868 exposure=310 title=SODIAL(R) 3000 Nail Art Gems Mixed Colours Shapes in Case (Size 2mm)
  - 612 cos=0.8853 exposure=83 title=5pcs Blue 2 Way Double Ended Nail Art Manicure Pedicure Dot Paint Dotting Painting Marbleizing Pen Tool
  - 9002 cos=0.8798 exposure=63 title=Bundle Monster 100 PC 3D Designs Nail Art Nailart Manicure Fimo Canes Sticks Rods Stickers Gel Tips
- Query 774 exposure=403 title=Olay Pro-X Advanced Cleansing System 0.68 Fl Oz, 1-Count
  - 2770 cos=0.6679 exposure=60 title=Olay Pro-X Replacement Brush Heads 2 Count
  - 1553 cos=0.5492 exposure=58 title=Olay Exfoliating Renewal Cleanser 6 Fl Oz
  - 5809 cos=0.5396 exposure=6 title=Eucerin Redness Relief Soothing Facial Moisture Lotion with SPF 15, 1.7 Oz
  - 566 cos=0.5321 exposure=130 title=Olay Regenerist Microdermabrasion &amp; Peel System 1 Kit
  - 4622 cos=0.5313 exposure=65 title=Queen Bee 100% All-Natural, Organic Under Eye Cream - Removes Dark Circles, Facial Lines and Wrinkles Naturally
  - 2079 cos=0.5234 exposure=247 title=Neutrogena Rapid Wrinkle Repair Eye, 0.5  Ounce
  - 2237 cos=0.5221 exposure=35 title=Olay Regenerist Detoxifying Pore Scrub 6.5 Fl Oz (Packaging May Vary) (Pack of 2)
  - 10002 cos=0.5198 exposure=7 title=Shiseido Benefiance Concentrated Anti-Wrinkle Eye Cream 15ml/0.51oz
  - 1965 cos=0.5110 exposure=14 title=Neutrogena Healthy Defense Daily Moisturizer, SPF 30, Light Tint 1.7 Ounce
  - 789 cos=0.5065 exposure=389 title=Aztec Secrets: Indian Healing Bentonite Clay, 2 lbs
- Query 278 exposure=391 title=World Pride Nail Tape Stripe Decoration Sticker Hologram, Set of 10
  - 301 cos=0.9602 exposure=321 title=15pcs Nail Art Painting Pen Brush
  - 277 cos=0.9460 exposure=282 title=MASH Rhinestones 2400 Piece 12 Color Nail Art Nailart Manicure Wheels
  - 300 cos=0.9401 exposure=431 title=Dotting 5 X 2 Way Marbleizing Dotting Pen Set for Nail Art Manicure Pedicure, 4 Ounce
  - 612 cos=0.9385 exposure=83 title=5pcs Blue 2 Way Double Ended Nail Art Manicure Pedicure Dot Paint Dotting Painting Marbleizing Pen Tool
  - 7842 cos=0.9370 exposure=79 title=Set of 3 Sable NAIL ART Brushes Pen, Detailer Liner and Striper
  - 2009 cos=0.9316 exposure=151 title=5pcs 2-ways Acrylic Uv Gel Nail Art Design Tips Dotting Painting Brush Pen Set
  - 294 cos=0.9289 exposure=113 title=BONAMART &reg; Premium MASH 100 Pc Nail Art Nailart 3d Manicure Design Sticks Rods Stickers Gel Tips
  - 292 cos=0.9216 exposure=310 title=SODIAL(R) 3000 Nail Art Gems Mixed Colours Shapes in Case (Size 2mm)
  - 295 cos=0.9155 exposure=305 title=Nail Art Brushes- Professional Nail Art Brushes- Sable Nail Art Brush Pen, Detailer, Liner **Set of 3
  - 9002 cos=0.9063 exposure=63 title=Bundle Monster 100 PC 3D Designs Nail Art Nailart Manicure Fimo Canes Sticks Rods Stickers Gel Tips
- Query 789 exposure=389 title=Aztec Secrets: Indian Healing Bentonite Clay, 2 lbs
  - 656 cos=0.6537 exposure=214 title=Thayers - Rose Petal Witch Hazel with Aloe Vera Alcohol-Free Toner - 12 oz.
  - 1358 cos=0.6486 exposure=121 title=NOW Solutions Castor Oil, 100 % Pure, 16 ounce
  - 4094 cos=0.6280 exposure=81 title=African Shea Butter Cream (100% Pure &amp; Raw, Gold) 8 Oz.
  - 2698 cos=0.6070 exposure=77 title=Dudu-Osun African Black Soap (100% Pure) 150g Pack of 4
  - 6901 cos=0.5966 exposure=24 title=Dudu-Osun African Black Soap (100% Pure) Pack of 3
  - 3961 cos=0.5711 exposure=10 title=Sumi Haigou Settuken Charcoal Bar Soap - 3 bars, 135g each
  - 4347 cos=0.5658 exposure=17 title=Lactic Acid 50% Gel Peel, 30ml (Professional)
  - 3210 cos=0.5652 exposure=5 title=Dudu Osun Black Soap 6 Pieces
  - 3212 cos=0.5580 exposure=91 title=Queen Helene Mint Julep Masque, 8 Ounce
  - 530 cos=0.5512 exposure=75 title=Wild Growth Hair Oil 4 Oz
- Query 861 exposure=329 title=Seche Vite Dry Fast Top Nail Coat, 0.5 Ounce
  - 6654 cos=0.7830 exposure=61 title=Seche Restore Nail Polish, 0.5 Fluid Ounce
  - 7759 cos=0.7696 exposure=34 title=Seche Vite Dry Fast Top Coat .5 Fl Oz
  - 1862 cos=0.7293 exposure=31 title=China Glaze Strong Adhesion Base Coat
  - 4062 cos=0.7269 exposure=44 title=Seche Restore, 2 Ounce
  - 1916 cos=0.7179 exposure=93 title=OPI Nail Envy Original, 0.5 Ounce
  - 10087 cos=0.7113 exposure=7 title=China Glaze Hologlam Holographic Galactic Gray
  - 8293 cos=0.7017 exposure=6 title=Orly Top 2 Bottom Basecoat-0.6 oz
  - 4964 cos=0.6948 exposure=35 title=Orly Nail Bonder Nail Treatment-0.6 oz
  - 2918 cos=0.6914 exposure=26 title=China Glaze Nail Polish, Jetstream, 0.5 Fluid Ounce
  - 8313 cos=0.6846 exposure=6 title=Color Club Covered In Diamonds 902 Nail Polish
- Query 94 exposure=328 title=Remington CI95AC/2 Tstudio Salon Collection Pearl Digital Ceramic Curling Wand, 1/2 Inch - 1 Inch
  - 365 cos=0.7289 exposure=189 title=Remington S9520 Salon Collection Ceramic Hair Straightener with Pearl Infused Wide Plates, 2 Inch
  - 52 cos=0.6803 exposure=66 title=Conair YOU CURL Curling Wand
  - 1097 cos=0.6765 exposure=110 title=Remington Ac2015 Tstudio Salon Collection Pearl Ceramic Hair Dryer
  - 3343 cos=0.6554 exposure=60 title=Neutrogena Wave Sonic Power Cleanser with 14 Foaming Pads
  - 4762 cos=0.6519 exposure=18 title=John Frieda Sleek Finish 1 Inch  Flat Iron
  - 1160 cos=0.6487 exposure=32 title=Infiniti Pro by Conair 1875 Watt Salon Performance Folding Handle Hair Dryer
  - 4783 cos=0.6470 exposure=56 title=Johnson's Baby Natural Shampoo, 10 Ounce (Pack of 2)
  - 2348 cos=0.6467 exposure=115 title=Simple Cleansing Facial Wipes, 25 Count (Pack of 2)
  - 1217 cos=0.6426 exposure=51 title=Aveeno Living Color Preserving Shampoo for Medium-Thick Hair, 10.5 Ounce (Pack of 2)
  - 1961 cos=0.6415 exposure=100 title=CoverGirl LashBlast Fusion Mascara Black 865, 1 Tube
- Query 301 exposure=321 title=15pcs Nail Art Painting Pen Brush
  - 278 cos=0.9602 exposure=391 title=World Pride Nail Tape Stripe Decoration Sticker Hologram, Set of 10
  - 277 cos=0.9398 exposure=282 title=MASH Rhinestones 2400 Piece 12 Color Nail Art Nailart Manicure Wheels
  - 612 cos=0.9358 exposure=83 title=5pcs Blue 2 Way Double Ended Nail Art Manicure Pedicure Dot Paint Dotting Painting Marbleizing Pen Tool
  - 2009 cos=0.9352 exposure=151 title=5pcs 2-ways Acrylic Uv Gel Nail Art Design Tips Dotting Painting Brush Pen Set
  - 300 cos=0.9279 exposure=431 title=Dotting 5 X 2 Way Marbleizing Dotting Pen Set for Nail Art Manicure Pedicure, 4 Ounce
  - 292 cos=0.9277 exposure=310 title=SODIAL(R) 3000 Nail Art Gems Mixed Colours Shapes in Case (Size 2mm)
  - 295 cos=0.9249 exposure=305 title=Nail Art Brushes- Professional Nail Art Brushes- Sable Nail Art Brush Pen, Detailer, Liner **Set of 3
  - 7842 cos=0.9204 exposure=79 title=Set of 3 Sable NAIL ART Brushes Pen, Detailer Liner and Striper
  - 5764 cos=0.9195 exposure=126 title=350buy Fashion Caviar Nails Art New 12 Colors plastic Beads Manicures or Pedicures Nail Art Hot Sales
  - 294 cos=0.9156 exposure=113 title=BONAMART &reg; Premium MASH 100 Pc Nail Art Nailart 3d Manicure Design Sticks Rods Stickers Gel Tips
- Query 292 exposure=310 title=SODIAL(R) 3000 Nail Art Gems Mixed Colours Shapes in Case (Size 2mm)
  - 286 cos=0.9544 exposure=238 title=Nail Art MoYou Silver Moon Rhinestone Pack of 1200 Crystal Premium Quality Gemstones in 12 different shapes and sizes, b
  - 5764 cos=0.9468 exposure=126 title=350buy Fashion Caviar Nails Art New 12 Colors plastic Beads Manicures or Pedicures Nail Art Hot Sales
  - 308 cos=0.9445 exposure=205 title=30Pcs Mixed Colors Rolls Striping Tape Line Nail Art Tips Decoration Sticker from Y2B
  - 295 cos=0.9444 exposure=305 title=Nail Art Brushes- Professional Nail Art Brushes- Sable Nail Art Brush Pen, Detailer, Liner **Set of 3
  - 301 cos=0.9277 exposure=321 title=15pcs Nail Art Painting Pen Brush
  - 1706 cos=0.9220 exposure=79 title=nail art powder 12color dust glitter sparkle nail tip decoration
  - 278 cos=0.9216 exposure=391 title=World Pride Nail Tape Stripe Decoration Sticker Hologram, Set of 10
  - 7745 cos=0.9198 exposure=52 title=1800pcs Nail Art Rhinestones Round 1.5mm
  - 612 cos=0.9183 exposure=83 title=5pcs Blue 2 Way Double Ended Nail Art Manicure Pedicure Dot Paint Dotting Painting Marbleizing Pen Tool
  - 3023 cos=0.9176 exposure=144 title=White Pearl Nail Art Stone Different Size Wheel Rhinestones Beads
- Query 295 exposure=305 title=Nail Art Brushes- Professional Nail Art Brushes- Sable Nail Art Brush Pen, Detailer, Liner **Set of 3
  - 292 cos=0.9444 exposure=310 title=SODIAL(R) 3000 Nail Art Gems Mixed Colours Shapes in Case (Size 2mm)
  - 286 cos=0.9443 exposure=238 title=Nail Art MoYou Silver Moon Rhinestone Pack of 1200 Crystal Premium Quality Gemstones in 12 different shapes and sizes, b
  - 308 cos=0.9434 exposure=205 title=30Pcs Mixed Colors Rolls Striping Tape Line Nail Art Tips Decoration Sticker from Y2B
  - 301 cos=0.9249 exposure=321 title=15pcs Nail Art Painting Pen Brush
  - 7842 cos=0.9226 exposure=79 title=Set of 3 Sable NAIL ART Brushes Pen, Detailer Liner and Striper
  - 6637 cos=0.9188 exposure=78 title=Nail Art MoYou Rhinestone Pack of 1200 Crystal Premium Quality 2mm Gemstones in 12 different colors, beauty accessory fo
  - 5764 cos=0.9181 exposure=126 title=350buy Fashion Caviar Nails Art New 12 Colors plastic Beads Manicures or Pedicures Nail Art Hot Sales
  - 277 cos=0.9170 exposure=282 title=MASH Rhinestones 2400 Piece 12 Color Nail Art Nailart Manicure Wheels
  - 278 cos=0.9155 exposure=391 title=World Pride Nail Tape Stripe Decoration Sticker Hologram, Set of 10
  - 612 cos=0.9150 exposure=83 title=5pcs Blue 2 Way Double Ended Nail Art Manicure Pedicure Dot Paint Dotting Painting Marbleizing Pen Tool
- Query 443 exposure=302 title=Opi Nail Lacquer, Not So Bora Pink, 0.5 Fluid Ounce
  - 1895 cos=0.8113 exposure=98 title=OPI Soft Shades Nail Lacquer, Princesses Rule!
  - 10187 cos=0.7905 exposure=5 title=OPI Nail Polish Vodka &amp; Caviar R55
  - 1896 cos=0.7904 exposure=5 title=Opi Hong Kong Collection Lucky Lucky Lavender Nl H48
  - 9233 cos=0.7877 exposure=8 title=OPI: Lacquer M37 Boyfriend Scales Lacquer, 0.5 oz OOS!!
  - 6290 cos=0.7846 exposure=13 title=OPI: Lacquer H55 Dutch Ya Just Love OPI? Lacquer, 0.5 oz
  - 447 cos=0.7837 exposure=18 title=Opi Texas Collection Don't Mess With Opi .5 oz.
  - 6923 cos=0.7835 exposure=21 title=OPI L00 Alpine Snow
  - 5757 cos=0.7800 exposure=12 title=OPI: Lacquer H59 Kiss Me On My Tulips Lacquer, 0.5 oz
  - 10097 cos=0.7764 exposure=7 title=OPI Nail Lacquer, I Vant to be A-Lone Star, 0.5 Fluid Ounce
  - 9701 cos=0.7741 exposure=15 title=Opi Nail Lacquer, Need Sunglasses, 0.5 Fluid Ounce

### Mid Exposure
- Query 4326 exposure=9 title=2 X 100G HESH BRAHMI POWDER
  - 210 cos=0.6777 exposure=8 title=Hesh Herbal Amla / Indian Gooseberry Powder For Dark &amp; Healthy Hair Naturally - 100 gms
  - 1605 cos=0.6699 exposure=54 title=Dabur Vatika Enriched Coconut Hair Oil 150ml (Pack of 2)
  - 7074 cos=0.6518 exposure=11 title=Elasta QP DPR-11+ Deep Penetrating Remoisturizing Conditioner Unisex 15 oz.
  - 5781 cos=0.6490 exposure=7 title=Jane Carter Nourish and Shine 4 oz &quot;Pack of 2&quot;
  - 5723 cos=0.6474 exposure=27 title=Jamaican Black Castor Oil Protein Hair Conditioner, 8oz
  - 8509 cos=0.6470 exposure=10 title=Professional Salon Hair Steamer with Rolling Floor Stand Base
  - 3622 cos=0.6430 exposure=43 title=Honeysuckle Rose Conditioner Aubrey Organics 11 oz Liquid
  - 6951 cos=0.6403 exposure=6 title=Tropic Isle Living Jamaican Black Castor Oil Hair Care Combo Set-II
  - 654 cos=0.6282 exposure=66 title=Tropic Isle Strong Roots Red Pimento Hair Growth Oil, 4 Ounce
  - 1370 cos=0.6280 exposure=36 title=Tropic Isle Jamaican Black Castor Oil Hair Food, 4 Ounce
- Query 10213 exposure=9 title=Ardell Brow &amp; Lash Growth Accelerator Treatment Gel 7ml/0.25oz
  - 5092 cos=0.5498 exposure=5 title=Physicians Formula Eye Definer Automatic Eye Pencil, Midnight Black 566
  - 7991 cos=0.5092 exposure=23 title=Urban Skintrition Prevention-Aid to fight Stretch Marks- 4 oz. /118 ml with Enhanced Shea Butter + Natural Botanical Ext
  - 2584 cos=0.4894 exposure=48 title=Slim Extreme 3D Thermo Active Cellulite Serum (8.8 oz)
  - 1886 cos=0.4778 exposure=61 title=RoC Multi-Correxion Night Treatment, 1 Ounce
  - 6425 cos=0.4672 exposure=5 title=Pantene Pro-V Split Fix 5.1 Fl Oz (Pack of 2)
  - 708 cos=0.4652 exposure=13 title=Olay Regenerist Filling + Sealing Wrinkle Treatment 1.0 Fl Oz
  - 5736 cos=0.4505 exposure=24 title=NIVEA Skin Firming and Toning Gel-Cream, 6.7 Ounce (Pack of 2)
  - 1540 cos=0.4481 exposure=18 title=Garnier Ultra-Lift Pro Gravity Defying Cream, Intensive, 1.5-Ounce
  - 5651 cos=0.4471 exposure=5 title=Slimquick Caffeine Free Clinical Strength, 72-count Box
  - 4200 cos=0.4446 exposure=58 title=GLAMGLOW Super-MudTM Clearing Treatment 1.2 oz
- Query 1475 exposure=9 title=Maybelline New York Dream Matte Powder, Sand, Medium 0-1, 0.32 Ounce
  - 1115 cos=0.6242 exposure=70 title=Maybelline New York The Colossal Volum' Express Washable Mascara, Glam Brown 232, 0.31 Fluid Ounce
  - 1772 cos=0.6175 exposure=36 title=Maybelline New York Expert Wear Eyeshadow Quads, Chai Latte 22q, 0.17 Ounce
  - 1321 cos=0.6042 exposure=44 title=Maybelline New York Dream Lumi Touch Highlighting Concealer, Ivory, 0.05 Fluid Ounce
  - 1477 cos=0.5922 exposure=39 title=NYX Cosmetics Concealer Wand, Nutmeg, 0.11  Oz
  - 1624 cos=0.5898 exposure=47 title=Maybelline New York Dream Liquid Mousse Foundation, Creamy Natural Light 5, 1 Fluid Ounce
  - 1116 cos=0.5885 exposure=66 title=Maybelline New York Fit Me! Foundation, 120 Classic Ivory, SPF 18, 1.0 Fluid Ounce
  - 2275 cos=0.5805 exposure=8 title=Maybelline New York Dream Matte Powder, Beige, Medium 2-2.5, 0.32 Ounce
  - 1196 cos=0.5783 exposure=8 title=Maybelline New York Expert Tools, Angled Definer Brush
  - 1200 cos=0.5746 exposure=80 title=Maybelline New York The Falsies Volum' Express Washable Mascara, Blackest Black, 0.25 Fluid Ounce
  - 1871 cos=0.5688 exposure=174 title=e.l.f. Studio Mineral Infused Face Primer
- Query 10203 exposure=9 title=Oil Control Facial Moisturizer: Lilac Stem Cells + 1% Chlorella Growth Factor - 1.75 Oz
  - 7948 cos=0.5513 exposure=10 title=ACURE Facial Cleansing Gel Superfruit + Chlorella, 4oz
  - 1649 cos=0.5318 exposure=15 title=Thayer Unscented Witch Hazel, 12 Fluid Ounce
  - 2240 cos=0.5176 exposure=5 title=Coppertone Oil Free Faces SPF 30, 3 Fluid Ounce
  - 3203 cos=0.5152 exposure=27 title=Lemongrass + Argan Stem Cell Conditioner - 8 oz - Liquid
  - 10047 cos=0.5054 exposure=8 title=NARS Shimmer Eyeshadow, Nepal
  - 8233 cos=0.5053 exposure=25 title=Utopia Care 5.5&quot; Professional Barber Razor Edge Hair Cutting Shears / Scissors with adjustable tension and finger i
  - 1000 cos=0.4979 exposure=21 title=humangear GoToob 2 Ounce (3 pack) Travel Bottle
  - 7947 cos=0.4965 exposure=7 title=Acure Organics Lemongrass + Moroccan Argan Oil Firming Body Lotion - 8 oz.
  - 6496 cos=0.4926 exposure=6 title=Lotion Ultra-Hydrating Cocoa Butter + CoQ10 Tube, 8oz
  - 508 cos=0.4914 exposure=38 title=Lemongrass + Argan Stem Cell Shampoo - 8 oz - Liquid
- Query 10197 exposure=9 title=Almay Smart Shade Makeup with SPF 15, Medium 300, 1 Ounce
  - 1304 cos=0.5330 exposure=12 title=Nail Tek Intensive Therapy II Buy 1 Get 1 free &quot;.5x2 oz&quot;
  - 6797 cos=0.5059 exposure=7 title=Konad Stamping Nail Art Image Plate M63
  - 9725 cos=0.5056 exposure=16 title=MASH Set of 25 Nail Art Nailart Polish Stamp Stamping Manicure Image Plates Accessories Set Kit
  - 2959 cos=0.4981 exposure=25 title=OPI Nail Lacquer, Pirates of The Caribbean Collection, Stranger Tides, 0.5 Fluid Ounce
  - 9090 cos=0.4954 exposure=44 title=Konad Stamping Nail Art Image Plate Holder, 0.6 Ounce
  - 6780 cos=0.4814 exposure=10 title=Revlon Beyond Natural Smoothing Primer, Clear, 0.85 Ounce
  - 1487 cos=0.4801 exposure=31 title=Avon GLIMMERSTICKS Eye Liner Cosmic Brown
  - 5545 cos=0.4791 exposure=26 title=Softsoap Lavender and Chamomile - Liquid Hand Soap Refill, 32 Ounce
  - 2326 cos=0.4723 exposure=7 title=Revlon Extra Life Top Coat, 0.5 Ounce
  - 8360 cos=0.4700 exposure=56 title=Konad Stamping Nail Art Image Plate - M57
- Query 10191 exposure=9 title=FASH Professional makeup Brush Set,12 pc, For Eye Shadow, Blush, Eyeliner,eyebrow....
  - 7070 cos=0.4947 exposure=32 title=Too Faced Cosmetics, Natural Eye, Neutral Eye Shadow Collection, 0.39 Ounce Net Wt.
  - 7155 cos=0.4935 exposure=6 title=Urban Decay Urban Ammo Eye Palette (Ammo Palette)
  - 2446 cos=0.4836 exposure=11 title=e.l.f. Brush Holder, Small
  - 104 cos=0.4833 exposure=247 title=NYX Cosmetics Eye Shadow Base, White, 0.21 Ounce
  - 12069 cos=0.4717 exposure=6 title=IBD Just Gel MOLLY Soak Off Neon Purple Nail Polish UV Manicure .5 oz Salon LED
  - 5825 cos=0.4677 exposure=19 title=Coastal Scents Go Makeup Palette, Beijing, 0.28 Oz
  - 7154 cos=0.4624 exposure=22 title=The Balm Balm Jovi New Holiday Palette for Lips, Eyes and Cheeks, .4 Ounce
  - 1124 cos=0.4611 exposure=34 title=bareMinerals Original Prime Time Foundation Primer
  - 10407 cos=0.4443 exposure=15 title=Palladio Herbal Foundation Primer, 0.674 Ounce
  - 10676 cos=0.4420 exposure=8 title=IBD Nail Lacquer, Whipped Cream, 0.5 Fluid Ounce
- Query 4894 exposure=9 title=American Crew Forming Cream, Medium Hold with Medium Shine, 3-Ounce Jars (Pack of 2) (Packaging may vary)
  - 4624 cos=0.6424 exposure=11 title=American Crew Pomade, 1.75 oz
  - 4893 cos=0.6012 exposure=8 title=Gold N Hot Gh2257 Professional 1875 Watt Ionic Dryer with Tourmaline
  - 977 cos=0.5663 exposure=57 title=Cetaphil Moisturizing Cream, 3.0 - Ounces Tube (Pack of 3)
  - 3196 cos=0.5622 exposure=38 title=Seki Edge Stainless Steel Fingernail Clipper
  - 4194 cos=0.5204 exposure=16 title=100% Unrefined Certified Grade A Shea Butter 4 oz.
  - 9428 cos=0.5201 exposure=6 title=American Crew Hair Stlying Pomade, 3 Ounce
  - 7202 cos=0.5140 exposure=5 title=The Shave Well Company Fog-Free Travel Mirror
  - 901 cos=0.4979 exposure=36 title=Seki Edge Toe Nail Clipper
  - 647 cos=0.4973 exposure=111 title=Biore  Deep Cleansing Pore Strips , 14 Nose Strips
  - 7201 cos=0.4876 exposure=8 title=St. Ives, Sensitive Skin Apricot Scrub, 6-Ounce (Pack of 6)
- Query 4886 exposure=9 title=Lime Crime Countessa Fluorescent Blue-Based Neon Pink Opaque Lipstick
  - 2676 cos=0.7740 exposure=17 title=Lime Crime Opaque Violet Purple Lipstick Airborne Unicorn
  - 10955 cos=0.7664 exposure=8 title=Lime Crime Chinchilla opaque grey lipstick
  - 2435 cos=0.7504 exposure=8 title=Lime Crime Great Pink Planet Opaque Barbie Pink Lipstick
  - 11914 cos=0.7440 exposure=5 title=Lime Crime My Beautiful Rocket Opaque Orange Lipstick
  - 1250 cos=0.7309 exposure=26 title=28 Color Neutral Warm Eyeshadow Palette Eye Shadow
  - 10149 cos=0.7246 exposure=9 title=Lime Crime Poisonberry Violet Purple Berry Lipstick
  - 6441 cos=0.7149 exposure=16 title=Lime Crime Opaque Nude Lipstick Coquette
  - 10721 cos=0.7081 exposure=5 title=Coastal Scents Creative Me #1 Makeup Palette
  - 9625 cos=0.6870 exposure=7 title=Ruby Kisses 24 HR Eyeshadow Magic Primer
  - 6538 cos=0.6201 exposure=19 title=Pro Beauty Makeup Sponge Blender Flawless Smooth Shaped Water Droplets Puff (Random Color)
- Query 5151 exposure=9 title=Yes to Grapefruit Even Skin Tone Moisturizer SPF 15, 1.4 Fluid Ounce
  - 2818 cos=0.8689 exposure=8 title=Yes To Carrots Scalp Relief Conditioner, 11.5 Fluid Ounce
  - 4952 cos=0.8101 exposure=8 title=Yes To Carrots Daily Facial Moisturizer SPF 15, 1.7 Fluid Ounce
  - 2817 cos=0.8045 exposure=9 title=Yes To Carrots Scalp Relief Shampoo, 11.5 Fluid Ounce
  - 5153 cos=0.8021 exposure=11 title=Yes To Volumizing Conditioner, Tomatoes, 16.9 Fluid Ounce
  - 5159 cos=0.7942 exposure=11 title=Yes To Tomatoes Blemish Clearing Facial Towelettes, 25 Count
  - 2077 cos=0.7837 exposure=9 title=Yes To Cucumber Daily Gentle Cleanser, 3.38 Fluid Ounce
  - 5150 cos=0.7822 exposure=11 title=Yes To Blueberries Intensive Skin Repair Serum, 1-Fluid Ounce
  - 5149 cos=0.7811 exposure=11 title=Yes To Blueberries Deep Wrinkle Night Cream, 1.7-Fluid Ounce
  - 4573 cos=0.7605 exposure=7 title=Yes To Deep Cleansing Facial Pads, Tomatoes, 50 Count
  - 5155 cos=0.7523 exposure=13 title=Yes To Tomatoes Daily Clarifying Cleanser, 3.38 Fluid Ounce
- Query 11817 exposure=9 title=Hair Repair Mask with Argan Oil and Shea Butter For Moisturizing Dry Damaged Hair-Great For Curly, Frizzy and Chemically
  - 12054 cos=0.9976 exposure=10 title=VITAMIN C SERUM 20% with Hyaluronic Acid For Your Face - The Best Topical Anti Aging Wrinkle Tightening Moisture Retenti
  - 11816 cos=0.9974 exposure=8 title=BEST Vitamin C Serum For-Face:: Organic &amp; Pure 20% Vitamin C + Hyaluronic Acid for Skin&#9733; #1 Anti Aging Beauty 
  - 11818 cos=0.9972 exposure=27 title=Face Cream - Anti Wrinkle Complex - Skin Care For AM/PM - Black Diamond Dust Infused - Beauty Skin Care Product - Skin R
  - 11820 cos=0.9929 exposure=15 title=Phytoceramides Anti Aging Supplement Reviews - Healthy Life Brand - Plant Derived 350 mg Skin Care Restoring Supplement 
  - 11814 cos=0.9906 exposure=6 title=Phytoceramides 350mg - Plant Derived Phytoceramide Is The Best Supplement To Take Ten Years Off Your Face Without Surger
  - 11855 cos=0.9893 exposure=11 title=Vitamin C Serum For YOUR skin- Face,Hands,Neck. Premium Salon-Grade L-Arginine, Hyaluronic Acid and Moisturizing Exotic 
  - 12055 cos=0.9889 exposure=12 title=BEST Cellulite Cream with Caffeine and Retinol - Body slimming and firming ***CONTAINS Dr. OZ RECOMMENDED ingredients***
  - 11821 cos=0.9888 exposure=14 title=Essential Makeup Brush Set - With Makeup Brush Holder - Leather Carrying Case 12 Professional Quality Brushes
  - 11815 cos=0.9880 exposure=6 title=Best Hyaluronic Acid Serum with Vitamin-C :: Top Rated 100 pure 1 oz Beauty Anti Aging Skin Care Cream and Moisturizer T
  - 9364 cos=0.9877 exposure=15 title=Dead Sea Mud Mask &#9733; 100% SATISFACTION GUARANTEED OR YOUR MONEY BACK &#9733; Brilliant Organic &amp; Natural Skin T

### Low Exposure
- Query 12070 exposure=5 title=600pc Crystal AB Round Rhinestone 3mm (10ss) 3D Acrylic Nail Art Decoration Cellphone Case USA SELLER! FAST SHIPPING! In
  - 9518 cos=0.8377 exposure=8 title=BMC UV LED Gel Nail Art Polish 3pc Kit One Color Red Top Base Coat Manicure Set
  - 10223 cos=0.8269 exposure=12 title=400X Lint Free Nail Art Wipes Acrylic Gel Tips Remover
  - 11679 cos=0.8183 exposure=5 title=BMC 6pc Color Gel Nail Art Polish UV LED Light Manicure Collection Set - NEONS, Stuck On You Like Gelly Collection
  - 11036 cos=0.7709 exposure=10 title=Nail Art Top Coat Topcoat + Primer Base Coat UV Gel Polish Gloss Guard Glaze Manicure Adhesives
  - 12033 cos=0.7583 exposure=5 title=Generic 12 Color Nail Art Dust Glitter Powder DIY Decoration Uv Acrylic Gel Tips
  - 11692 cos=0.7440 exposure=10 title=IBD Bonder
  - 11428 cos=0.7417 exposure=7 title=IBD Just Gel INFINITELY CURIOUS Soak Off Orange Nail Polish UV Manicure .5oz LED
  - 8379 cos=0.7415 exposure=51 title=French, Chevron &amp; Teardrop Nail Tip Guides Stickers (Pack of 5)
  - 11474 cos=0.7362 exposure=13 title=Liquid Basic Starter Kit Acrylic-pulver Acrylic Powder Brush for 3D Nail Art Full Set
  - 3409 cos=0.7355 exposure=54 title=10PC wearable nail art soakers Ongle acrylic removal
- Query 12073 exposure=5 title=MapofBeauty 28&quot; 70cm Long Curly Hair Ends Costume Cosplay Wig (Brown)
  - 11260 cos=0.9297 exposure=7 title=Cosplay Inshop 26&quot; 65cm Wavy Gradient Black to Diamond Green Lolita Costume Cosplay Wig
  - 8814 cos=0.9219 exposure=12 title=32&quot; 80cm Spiral Curly Cosplay Wig--Light Blonde
  - 10143 cos=0.9171 exposure=6 title=MapofBeauty Long Wave Curly Hair Wig Full Wig for Women Long (Black)
  - 6470 cos=0.8970 exposure=17 title=32&quot; 80cm Long Hair Heat Resistant Spiral Curly Cosplay Wig (Red Dark)
  - 11342 cos=0.8961 exposure=5 title=SureWells Vocaloid Series Silver White Long Curly Cosplay Wig Costume Wigs
  - 9076 cos=0.8921 exposure=6 title=Sexy Women Wigs Lace Wigs Oblique Bangs Medium for Women Hair Wigs Wigs Store
  - 4517 cos=0.8845 exposure=16 title=Black Wig Stand, Portable Wig Stand, Wig Dryer, 6P
  - 6428 cos=0.8828 exposure=5 title=K-on! Kotobuki Tsumugi Anime Cosplay Wig (Model:jf010067)
  - 10776 cos=0.8801 exposure=5 title=X&amp;Y ANGEL New Short Curly Synthetic Hair Fiber Wigs Toupee M123
  - 5204 cos=0.8754 exposure=38 title=MelodySusie Beautiful Long Dark Brown Curly Wave Stunning Wig Full Wig + MelodySusie Wig Cap + MelodySusie Wig Comb
- Query 12074 exposure=5 title=350buy NEW 100X Sparkling French Nail Tips Stunning Mix Glitter Colors Style False French Acrylic Nail Art Tips
  - 11706 cos=0.9608 exposure=13 title=350buy 24 Colors 3D Nail Art Glitter Acrylic Powder Decoration
  - 12038 cos=0.9371 exposure=8 title=350buy 36 Acrylic Powder Liquid KITS Primer UV NAIL ART TIP Set Dust Stickers Brush
  - 11543 cos=0.9343 exposure=5 title=350BUY Sweet Clear Pink Hearts Acrylic French False Nail Art Tips 50pcs
  - 11420 cos=0.9326 exposure=13 title=Clear Crystal Acrylic Powder for Acrylic Liquid Nail Art Tips
  - 12049 cos=0.9325 exposure=7 title=24 Color Acrylic Powder Dust Nail Art Decoration
  - 7846 cos=0.9324 exposure=15 title=5 Pcs Acrylic UV Gel Nail Art Flase Practice Finger Training Display Decoration Tool
  - 7921 cos=0.9267 exposure=17 title=10pcs nail art silver foil reusable acrylic UV gel forms shape french false Silver
  - 7922 cos=0.9266 exposure=25 title=350BUY Clear White Pink Acrylic Powder Builder for Nail Art Manicure High Quality
  - 11198 cos=0.9264 exposure=38 title=500 Pcs Clear French Acrylic Style Artificial Half False Nails Nail Art Tips
  - 11700 cos=0.9237 exposure=18 title=niceEshop 500 French Acrylic False Artificial Tips Nail Art -- Pink
- Query 12076 exposure=5 title=Set 5 Wheels of Nail Art Fimo Slices Decal 3d Decorations 60 Designs by Cheeky&reg;
  - 12020 cos=0.9027 exposure=10 title=10pcs Cute 3D Design Nail Art Nailart Manicure Animal Pattern Fimo Canes Sticks Rods Stickers Gel Tips Decoration
  - 12022 cos=0.8826 exposure=7 title=60X Dried Flower 3D Nail Art Decoration UV Gel Acrylic
  - 11571 cos=0.8765 exposure=5 title=Jovivi 5 X 2 Way Nail Art Manicure Pedicure Dot Paint Dotting Painting Marbleizing Tools Pen Set
  - 11572 cos=0.8674 exposure=9 title=144pcs Fimo Slice Lovely Animal Nail Art Decoration
  - 278 cos=0.8634 exposure=391 title=World Pride Nail Tape Stripe Decoration Sticker Hologram, Set of 10
  - 860 cos=0.8606 exposure=71 title=Bundle 5 Wheels Premium Manicure Nail Art Decorations Total 7400 Gems By Cheeky&reg;
  - 10340 cos=0.8592 exposure=11 title=3000pcs 2mm 12 Color Nail Art Nailart Heart Shape Rhinestones Glitter Tips Decoration + Wheel
  - 5764 cos=0.8579 exposure=126 title=350buy Fashion Caviar Nails Art New 12 Colors plastic Beads Manicures or Pedicures Nail Art Hot Sales
  - 11712 cos=0.8574 exposure=7 title=240 pcs FIMO Nail Art Miniature Decorations with Wheel
  - 301 cos=0.8548 exposure=321 title=15pcs Nail Art Painting Pen Brush
- Query 12077 exposure=5 title=Gelish - The Shadows Collection - The Perfect Silhouette # 01460
  - 11725 cos=0.8502 exposure=6 title=OPI Gelcolor Collection Nail Gel Lacquer, Lincoln Park After Dark, 0.5 Fluid Ounce
  - 12005 cos=0.8407 exposure=7 title=OPI Gelcolor Collection Nail Gel Lacquer, Strawberry Margarita, 0.5 Fluid Ounce
  - 3385 cos=0.8370 exposure=15 title=Gelish Soak-Off Gel Polish by Nail Harmony - 01600 Wiggle Fingers Wiggle Thumbs
  - 6500 cos=0.8287 exposure=29 title=Nail Harmony Gelish Soak Off Gel Polish - Hot Rod Red (15ml / .5 Oz) - 01412
  - 7766 cos=0.8270 exposure=29 title=Harmony Foundation Base, Top if off and ph Bond - 3 PACK
  - 10980 cos=0.8183 exposure=12 title=Gelish U V Gel Nail Polish &quot;Sea Foam&quot; #01341
  - 11781 cos=0.8178 exposure=9 title=Gelish - House of Gelish Collection - My Favorite Accessory #01439
  - 11511 cos=0.8073 exposure=6 title=Harmony Gelish Sizzling Summer Nights Collection - Showstopping .5 oz.
  - 7058 cos=0.8068 exposure=11 title=Harmony Gelish UV Soak Off Gel Polish Elegant Wish
  - 11962 cos=0.8020 exposure=6 title=Gelish Brights Have More Fun - #01557
- Query 12083 exposure=5 title=Best Anti Aging Cream Reduces Wrinkels in Women and Men - Clinical Strength Bio-Peptide Wrinkle Cream Reduces Deep Wrink
  - 12081 cos=0.9943 exposure=10 title=Phytoceramides - Plant Derived Phytoceramides with Vitamins A, C, D and E to Moisturize &amp; Rejuvenate Skin - Replenis
  - 12078 cos=0.9916 exposure=17 title=Best Vitamin C Serum For Your Face Contains Vitamin C + E + Hyaluronic Acid Serum - Best Pure Organic Potent 1oz Bottle 
  - 12082 cos=0.9883 exposure=9 title=Best Anti Aging Cream Reduces Wrinkels in Women and Men - Clinical Strength Bio-Peptide Wrinkle Cream Reduces Deep Wrink
  - 11769 cos=0.9875 exposure=16 title=Moroccan Magic Argan Oil Serum - SALE 50% OFF TODAY - #1 Salon Quality Formula Hair Treatment Product That Will Conditio
  - 12079 cos=0.9819 exposure=27 title=Biotin Vitamin Supplement Benefits Skin, and Promotes Healthy Hair and Nail Growth and Weight Loss 5,000 mcg
  - 11853 cos=0.9780 exposure=33 title=Vitamin-C Serum - Potent 20% Topical Vitamin C + E + A Serum - Best Anti-aging + Anti-wrinkle Skin Care Combination Eras
  - 4034 cos=0.9583 exposure=10 title=THAT Eye Cream All-In-One Eye Gel with Vitamin C - Best Anti Aging Serum For Wrinkles, Dark Circles, Puffiness, Bags and
  - 10257 cos=0.9464 exposure=30 title=Seaweed Powder - The Best Cellulite Treatment - Cellulite Remover PowerHouse - Pure Ascophyllum Nodosum Kelp Powder to b
  - 11766 cos=0.9384 exposure=10 title=Buy Extra Virgin Coconut Oil; 16 Oz.; Health Benefits; Great for Hair, Skin, Weight Loss, Many Benefits, Cook with It; O
  - 12053 cos=0.9258 exposure=27 title=Organic Argan Oil Young for Life - Certified, Pure, Natural, Virgin Moroccan Cold Pressed - For Your Hair, Skin, Face, N
- Query 12084 exposure=5 title=Essence of Argan Oil 50ml
  - 12087 cos=1.0000 exposure=5 title=Essence of Argan Conditioner (100% Organic)
  - 12088 cos=1.0000 exposure=5 title=Argan Oil Soap Essence of Argan
  - 12089 cos=1.0000 exposure=5 title=Essence of Argan Hair Masque
  - 12086 cos=1.0000 exposure=5 title=Essence of Argan Shampoo (100% Organic)
  - 12085 cos=1.0000 exposure=5 title=Essence of Argan Pure Morrocan Oil 15ML (100% Organic)
  - 7781 cos=0.9617 exposure=5 title=Mary Kay Oil Free Eye Makeup Remover 3.75 fluid ounce
  - 11160 cos=0.7223 exposure=5 title=FreeTress Equal FUTURA Hair Wide Lace Front Wig - NELLY (Deep Invisible Part) (OP61327)
  - 11489 cos=0.7157 exposure=8 title=Freetress Equal Lace Front Baby Hairline Wig - Abby - 1
  - 11180 cos=0.6673 exposure=8 title=Freetress Equal Lace Front Natural Hairline Wig - Estelle-1B
  - 41 cos=0.6389 exposure=21 title=Mary Kay Mineral Powder Foundation Beige 2
- Query 12085 exposure=5 title=Essence of Argan Pure Morrocan Oil 15ML (100% Organic)
  - 12086 cos=1.0000 exposure=5 title=Essence of Argan Shampoo (100% Organic)
  - 12084 cos=1.0000 exposure=5 title=Essence of Argan Oil 50ml
  - 12087 cos=1.0000 exposure=5 title=Essence of Argan Conditioner (100% Organic)
  - 12088 cos=1.0000 exposure=5 title=Argan Oil Soap Essence of Argan
  - 12089 cos=1.0000 exposure=5 title=Essence of Argan Hair Masque
  - 7781 cos=0.9628 exposure=5 title=Mary Kay Oil Free Eye Makeup Remover 3.75 fluid ounce
  - 11160 cos=0.7229 exposure=5 title=FreeTress Equal FUTURA Hair Wide Lace Front Wig - NELLY (Deep Invisible Part) (OP61327)
  - 11489 cos=0.7165 exposure=8 title=Freetress Equal Lace Front Baby Hairline Wig - Abby - 1
  - 11180 cos=0.6683 exposure=8 title=Freetress Equal Lace Front Natural Hairline Wig - Estelle-1B
  - 41 cos=0.6391 exposure=21 title=Mary Kay Mineral Powder Foundation Beige 2
- Query 12016 exposure=5 title=100pcs Zebra Pattern French Artificial Half False Nails Nail Art Tips + Glue + Box #D2
  - 7881 cos=0.8456 exposure=5 title=100 Pcs Black Acrylic French Style Artificial Half False Nails Nail Art Tips + Box + Glue
  - 11198 cos=0.8269 exposure=38 title=500 Pcs Clear French Acrylic Style Artificial Half False Nails Nail Art Tips
  - 11543 cos=0.8259 exposure=5 title=350BUY Sweet Clear Pink Hearts Acrylic French False Nail Art Tips 50pcs
  - 11705 cos=0.8225 exposure=8 title=STAR NAIL Natural Nail Dehydrant 1 oz.
  - 11739 cos=0.8082 exposure=5 title=3pcs Marble Sable Acrylic Tips Nail Art Painting Brush Brushes Carving Pen Detachable Size #2 #6 #8
  - 11663 cos=0.8012 exposure=7 title=5PC nail art practice finger acrylic display Ongle
  - 10914 cos=0.7987 exposure=6 title=500 White French False Acrylic Nail Art Tips Gel Makeup
  - 11478 cos=0.7942 exposure=8 title=Acrylic Powder Kits Primer Basis Nail Art DIY Decoration Set Kit w/ Brush
  - 7809 cos=0.7936 exposure=16 title=Supernail Nail Primer, 0.25 Fluid Ounce
  - 11738 cos=0.7932 exposure=12 title=5pcs Nail Art Sable Acrylic Nail Brushes Pen Set with Cuticle Pusher Ends, 5 Sizes
- Query 12018 exposure=5 title=PRO IRON CLIPPER for ACRYLIC/FALSE NAILS TIP CUTTER
  - 7141 cos=0.7081 exposure=18 title=KDS Nail Glue 10-pk.
  - 1682 cos=0.7012 exposure=51 title=500pcs Lady White French Acrylic Style Artificial False Nails Half Tips
  - 9855 cos=0.6908 exposure=14 title=MelodySusie Nails Art Professional Manicure Electric Nail Drill File + MelodySusie Nail Nipper
  - 10872 cos=0.6786 exposure=5 title=Q-Pink Cutters (C-Shape Regular)
  - 11736 cos=0.6779 exposure=12 title=Crystal Glass Dappen Dish with Stainless Steel Lid
  - 12017 cos=0.6727 exposure=7 title=USpicy Electric Pen-Shape Nail Drill with 1 AC adapter and 6 pcs Bit Acrylic UV GEL
  - 9892 cos=0.6586 exposure=7 title=Pro Gold Small Carbide File. Small Barrel. Coarse Bit. Fits 3/32&quot;
  - 10315 cos=0.6583 exposure=33 title=500 WHITE French Acrylic False Artificial Tips Nail Art
  - 7142 cos=0.6583 exposure=18 title=Clear Tip 550pc/box, Box Include
  - 6631 cos=0.6534 exposure=11 title=10 Wheels Premium Manicure Nail Art Decorations Total of 15000 Gems By Cheeky&reg;

## Summary JSON

```json
{
  "source": "LETTER Beauty.inter.json",
  "method": "item-item cooccurrence PPMI + TruncatedSVD",
  "interaction_format": "dict user_id -> list[item_id] or dict with sequence-like fields",
  "num_items": 12101,
  "embedding_dim": 128,
  "window_size": 5,
  "cooc_nnz": 1050000,
  "cooc_sum": 682862.5,
  "ppmi_nnz": 1043112,
  "ppmi_sum": 4145201.0,
  "item_order_aligned": true,
  "skipped_items": 0,
  "row_norm_mean": 1.0,
  "row_norm_median": 1.0,
  "row_norm_min": 0.9999997615814209,
  "row_norm_max": 1.0000001192092896,
  "zero_row_count": 0,
  "zero_row_indices_first_20": [],
  "svd_explained_variance_sum": 0.11419887095689774,
  "cf_cluster_n_clusters": 256,
  "cf_cluster_empty_count": 0,
  "cf_cluster_min_size": 7,
  "cf_cluster_max_size": 134,
  "cf_cluster_median_size": 46.0,
  "cf_cluster_neighbor_observed_sharing_rate": 0.2438244795303709,
  "cf_cluster_neighbor_random_sharing_rate": 0.004990376918229353,
  "cf_cluster_neighbor_lift": 48.858930602957905,
  "cf_cluster_neighbor_observed_pair_count": 176139,
  "cf_cluster_vs_original_c1_nmi": 0.4237630636292303,
  "cf_cluster_vs_original_c1_purity": 0.3269151309809107,
  "cf_cluster_vs_product_type_nmi": 0.2631732516665439,
  "cf_cluster_vs_product_type_purity": 0.3131146186265598,
  "valid": true,
  "recommended_for_cr_letter_l_cf": true,
  "warnings": [],
  "outputs": {
    "embedding": "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_item_emb.npy",
    "item_id_order": "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_item_id_order.json",
    "cluster_labels": "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_cluster_labels.npy",
    "cluster_centers": "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_cluster_centers.npy",
    "summary": "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_embedding_summary.json",
    "report": "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/reports/Beauty_cf_svd_embedding_report.md"
  }
}
```
