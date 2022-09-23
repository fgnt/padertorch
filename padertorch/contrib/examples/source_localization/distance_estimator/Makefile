.DEFAULT_GOAL = complete

# Add values for missing constants here or pass it via the terminal

# path where the RIR database should be stored
RIR_PATH =
# path where the JSON of the RIR database should be stored (optional, otherwise stored in the same directory as the RIR database)
RIR_JSON =
# path where the LibriSpeech database should be stored, can be left empty if the database already exists
LIBRI_PATH =
# path where the JSON of the LibriSpeech database should be stored (optional, if LibriSpeech is not already existing, otherwise stored in the same directory as LibriSpeech)
LIBRI_JSON =
# path where the JSON containing the VAD information for LibriSpeech should be stored (optional, unless LibriSpeech gets updated)
VAD_JSON =

complete: #prepare all perequisites
ifndef RIR_PATH
	$(error 'Please specify RIR_PATH; the path, where the RIR database should be stored or fill out the constants in the Makefile')
endif
ifndef LIBRI_PATH
	$(error 'Please specify LIBRI_PATH; the path, where the LibriSpeech database should be stored or fill out the constants in the Makefile')
endif
ifndef VAD_JSON
	$(eval VAD_JSON=$(LIBRI_PATH))
endif
	python download.py with rir_path=$(RIR_PATH) libri_path=$(LIBRI_PATH) vad_json_path=$(VAD_JSON)
#since tar_info of LibriSpeech creates an own LibriSpeech subdirectory in the specified path, add this to the path before the JSON is created
	$(eval override LIBRI_PATH=$(addsuffix /LibriSpeech,$(LIBRI_PATH)))
	python create_jsons.py with rir_path=$(RIR_PATH) rir_json_path=$(RIR_JSON) libri_path=$(LIBRI_PATH) libri_json_path=$(LIBRI_JSON) vad_json_path=$(VAD_JSON)

rir: #assumes that LibriSpeech already exists
ifndef LIBRI_JSON
	$(error 'Please specify LIBRI_JSON; the path, where the LibriSpeech JSON is stored or fill out the constants in the Makefile')
endif
ifndef VAD_JSON
	$(error 'Please specify VAD_JSON; the path, where the JSON with VAD information should be stored or fill out the constants in the Makefile')
endif
ifndef RIR_PATH
	$(error 'Please specify RIR_PATH; the path, where the RIR database should be stored or fill out the constants in the Makefile')
endif
	python download.py with rir_path=$(RIR_PATH) vad_json_path=$(VAD_JSON)
	python create_jsons.py with update_librispeech rir_path=$(RIR_PATH) rir_json_path=$(RIR_JSON) libri_json_path=$(LIBRI_JSON) vad_json_path=$(VAD_JSON)


librispeech_full: # assumes that the RIR database already exists
ifndef LIBRI_PATH
	$(error 'Please specify LIBRI_PATH, the path, where the LibriSpeech database should be stored or fill out the constants in the Makefile')
endif
ifndef VAD_JSON
	$(eval VAD_JSON=$(LIBRI_PATH))
endif
	python download.py with libri_path=$(LIBRI_PATH) vad_json_path=$(VAD_JSON)
#since tar_info of LibriSpeech creates an own LibriSpeech subdirectory in the specified path, add this to the path before the JSON is created
	$(eval override LIBRI_PATH=$(addsuffix /LibriSpeech,$(LIBRI_PATH)))
	python create_jsons.py with libri_path=$(LIBRI_PATH) libri_json_path=$(LIBRI_JSON) vad_json_path=$(VAD_JSON)

librispeech_update: #assumes that both databases exist, therefore only LibriSpeech must be updated with VAD information if LibriSpeech should be used for speech source signals.
ifndef LIBRI_JSON
	$(error 'Please specify LIBRI_JSON; the path, where the LibriSpeech-JSON is stored or fill out the constants in the Makefile')
endif
ifndef VAD_JSON
	$(error 'Please specify VAD_JSON; the path, where the JSON with VAD information should be stored or fill out the constants in the Makefile')
endif
	python download.py with vad_json_path=$(VAD_JSON)
	python create_jsons.py with update_librispeech libri_json_path=$(LIBRI_JSON) vad_json_path=$(VAD_JSON)
	
	

