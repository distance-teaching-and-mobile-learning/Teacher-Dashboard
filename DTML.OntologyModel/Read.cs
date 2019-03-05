        public static void ImportModel(string protobufFile="/Data/adj.protobuf")
        {
            using (FileStream fso = File.OpenRead(protobufFile))
            {
                var crcFile = ProtoBuf.Serializer.Deserialize<Graph>(fso);
                fso.Close();
            }

            Console.Write("Done.. Deserialized");
          }