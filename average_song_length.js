db.getCollection("songs").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$match: { 
			        "$and" : [
			            {
			                "_id" : {
			                    "$gte" : NumberInt(8273)
			                }
			            }, 
			            {
			                "_id" : {
			                    "$lte" : NumberInt(8359)
			                }
			            }
			        ]
			    }
		},

		// Stage 2
		{
			$group: {
			"_id": null, 
			"avgSongLength": { "$avg": "$track_length_s" },
			}
		},

	]

	// Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/

);
