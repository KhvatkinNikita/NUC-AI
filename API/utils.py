def lim2id(geo_lim):
    """
    geo_lim to geo_id converter for REE API, as it needs the ID to work
    geo_lim [str]: string identifying the Autonomous Community
     
    """
    # general map
    general = ['peninsular', 'canarias', 'baleares', 'ceuta', 'melilla']
    general_id = list(range(8741, 8746))
    general_map = dict(zip(general, general_id))
    
    # ccaa map
    ccaa = [
        'Andalucía', 'Aragón', 'Cantabria', 'Castilla la Mancha', 'Castilla y León',
        'Cataluña', 'País Vasco', 'Principado de Asturias', 'Comunidad de Madrid',
        'Comunidad de Navarra', 'Comunidad Valenciana', 'Extremadura', 'Galicia',
        'Islas Baleares', 'Islas Canarias', 'La Rioja', 'Región de Murcia'
    ]
    
    ccaa_id = list(range(4, 22))
    ccaa_id.remove(12)  # Remove 12, because REE does not have it 
    ccaa_map = dict(zip(ccaa, ccaa_id))

    # raise error if not valid input
    valid_options = list(general_map.keys()) + list(ccaa_map.keys())
    if geo_lim not in valid_options:
        raise ValueError(f"Invalid input: '{geo_lim}'. Valid options are: {valid_options}")
    
    return general_map.get(geo_lim, ccaa_map.get(geo_lim, None))