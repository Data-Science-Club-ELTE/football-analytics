
import ScraperFC as sfc


ss = sfc.Sofascore()


#eto_heat = ss.scrape_heatmaps(14019475)
#eto_momentum = ss.scrape_match_momentum(14019475)
eto_shots = ss.scrape_match_shots(14419287)
#eto_match = ss.scrape_team_match_stats(14019475)
#et = ss.get_match_dict(14019475)
#print(eto_heat)
#print(eto_momentum)
print(eto_shots)
#print(eto_shots.columns)
#print(et)
eto_shots.to_excel("shots.xlsx")