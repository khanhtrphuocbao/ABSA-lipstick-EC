import vn.pipeline.Annotation;
import vn.pipeline.Sentence;
import vn.pipeline.VnCoreNLP;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Base64 framing keeps each review independent, including multiline/punctuation-only input. */
class VnCoreNLPBatch {
    public static void main(String[] args) throws Exception {
        if(args.length!=2)throw new IllegalArgumentException("input TSV, output TSV required");
        VnCoreNLP pipeline=new VnCoreNLP(new String[]{"wseg"});
        try(BufferedReader in=Files.newBufferedReader(Path.of(args[0]),StandardCharsets.UTF_8);
            BufferedWriter out=Files.newBufferedWriter(Path.of(args[1]),StandardCharsets.UTF_8)) {
            String line;int count=0;
            while((line=in.readLine())!=null) {
                String[] fields=line.split("\t",-1);
                if(fields.length!=2)throw new IOException("Invalid framed record");
                String text=new String(Base64.getDecoder().decode(fields[1]),StandardCharsets.UTF_8);
                Annotation annotation=new Annotation(text);
                pipeline.annotate(annotation);
                List<String> sentences=new ArrayList<>();
                if(annotation.getSentences()!=null)
                    for(Sentence s:annotation.getSentences())sentences.add(s.getWordSegmentedSentence());
                String segmented=String.join(" ",sentences).trim();
                out.write(fields[0]+"\t"+Base64.getEncoder().encodeToString(segmented.getBytes(StandardCharsets.UTF_8))+"\n");
                if(++count%2000==0)System.err.println("Segmented reviews: "+count);
            }
        }
    }
}
