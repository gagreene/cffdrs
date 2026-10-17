"""Shared default seasonal grass curing values."""
def getSeasonGrassCuring(season: str,
                         province: str,
                         subregion: str | None = None) -> int | None:
    """
    Function returns a default grass curing code based on season, province, and subregion
    :param season: annual season ("spring", "summer", "fall", "winter")
    :param province: province being assessed ("AB", "BC")
    :param subregion: British Columbia subregions ("southeast", "other")
    :return: grass curing percent (%); "None" is returned if season or province are invalid
    """
    if province == 'AB':
        # Default seasonal grass curing values for Alberta
        # These curing rates are recommended by Neal McLoughlin to align with Alberta Wildfire knowledge and practices
        gc_dict = {
            'spring': 75,
            'summer': 40,
            'fall': 60,
            'winter': 100
        }
    elif province == 'BC':
        if subregion == 'southeast':
            # Default seasonal grass curing values for southeastern British Columbia
            gc_dict = {
                'spring': 100,
                'summer': 90,
                'fall': 90,
                'winter': 100
            }
        else:
            # Default seasonal grass curing values for British Columbia
            gc_dict = {
                'spring': 100,
                'summer': 60,
                'fall': 85,
                'winter': 100
            }
    else:
        gc_dict = {}

    return gc_dict.get(season.lower(), None)



