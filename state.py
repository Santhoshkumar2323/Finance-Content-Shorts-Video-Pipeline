from typing import TypedDict, Optional, List, Annotated

class Beat(TypedDict, total=False):
    index: int                     
    text: str                      
    image_prompt: str             

    audio_path: Optional[str]     
    audio_duration: Optional[float]
    image_path: Optional[str]      

def merge_beats(current: List[Beat], update: List[Beat]) -> List[Beat]:
    if not current:
        return update
    merged = []
    for existing_beat, updated_beat in zip(current, update):
        combined = dict(existing_beat)
        for key, value in updated_beat.items():
            if value is not None:
                combined[key] = value
        merged.append(combined)
    return merged


class PipelineState(TypedDict, total=False):
    raw_input: str                
    beats: Annotated[List[Beat], merge_beats]  
    validation_passed: bool        
    validation_attempts: int       
    final_video_path: Optional[str]  
    run_id: str                   
                                   